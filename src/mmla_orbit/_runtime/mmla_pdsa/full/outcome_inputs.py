"""Immutable ingress for the already materialized finite training course.

No tokenizer/model/environment is called. No truncation, new examples, source
selection, retokenization, offline feedback substitution or test access occurs.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib,json
import torch
from .alfworld_episode import load_permits,TaskPermit,ReplayEpisode,_digest,_sha
from .alfworld_formatting import SupervisedText
from .outcome_schedule import make_schedule,SEEDS
from .outcome_session import OutcomeSession
from .outcome_trajectory import OutcomeTrajectoryCore

PARTITION='a0a5a4b8f163785f599a2271e0e3c1c46ab0b3801d9289a79cbecfc4062e7e11'
GENERAL='0a1765c8221b8edc1adc2b3efc9d97a389a00cdbf4a59fd376d722852e2b0ba1'
SCHEDULE='745f2a7aea445f8a10b7492baf793ea986efe1cf7002b8a05024b25cee7d2acd'


def _pairs(rows):
    result={}
    for key,value in rows:
        if key in result:raise ValueError('duplicate JSON key')
        result[key]=value
    return result


def _json(data):
    return json.loads(data,object_pairs_hook=_pairs,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def read_general(path,*,expected_sha256,tokenizer_identity,expected_rows):
    _sha(expected_sha256);_sha(tokenizer_identity)
    if type(expected_rows) is not int or not 1<=expected_rows<=7890:raise ValueError('fixed finite row count')
    path=Path(path)
    if path.stat().st_size>128*1024**2:raise ValueError('general file exceeds fixed input envelope')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_sha256:raise ValueError('general materialization identity changed')
    identity=_digest(dict(schema='mmla-general-chat-ce-v1',file=expected_sha256,tokenizer=tokenizer_identity,
        labels='original-multi-assistant-target-mask',max_original_tokens=2048))
    rows={};sources=set()
    required={'schema','source_sha256','shard','index','source','input_ids','target_mask','chat_sha256'}
    for line in data.splitlines():
        row=_json(line)
        if set(row)!=required or row['schema']!='mmla-general-chat-ce-v1':raise ValueError('exact original general schema')
        source=_sha(row['source_sha256']);key=_sha(row['chat_sha256'])
        if key in rows or source in sources:raise ValueError('duplicate general identity')
        sources.add(source)
        ids,mask=row['input_ids'],row['target_mask']
        if (type(ids) is not list or type(mask) is not list or not 1<len(ids)<=2048 or len(mask)!=len(ids)
                or any(type(i) is not int or not 0<=i<130560 for i in ids)
                or any(type(m) is not int or m not in (0,1) for m in mask)
                or mask[0]!=0 or not any(mask)):
            raise ValueError('original full token sequence and assistant mask required')
        first=mask.index(1)
        rows[key]=SupervisedText(tuple(ids),tuple(i if m else -100 for i,m in zip(ids,mask)),
            first,identity,'general-multi-assistant')
    if len(rows)!=expected_rows:raise ValueError('complete fixed general population required')
    return MappingProxyType(rows)


@dataclass(frozen=True)
class ScheduledTask:
    permit:TaskPermit
    environment_seed:int
    fast_seed:int

@dataclass(frozen=True)
class TrainingBatch:
    objective:str
    seed:int
    update:int
    tasks:tuple[ScheduledTask,...]
    general_ids:tuple[str,...]
    general:tuple[SupervisedText,...]
    schedule_identity:str


class FiniteInputs:
    def __init__(self,data_root,*,tokenizer_identity):
        root=Path(data_root)
        permits=load_permits(root/'task-partition.jsonl',expected_sha256=PARTITION)
        general=read_general(root/'general-train-tokens.jsonl',expected_sha256=GENERAL,
            tokenizer_identity=tokenizer_identity,expected_rows=7890)
        data=(root/'finite-schedule-r1.json').read_bytes()
        if hashlib.sha256(data).hexdigest()!=SCHEDULE:raise ValueError('original finite course changed')
        actual=_json(data);expected=make_schedule(permits,tuple(general))
        if _digest(actual)!=_digest(expected):raise ValueError('persisted course differs from fixed result-blind procedure')
        self.permits=MappingProxyType(permits);self.general=general
        self.evaluation=MappingProxyType({key:tuple(value) for key,value in actual['evaluation'].items()})
        self._courses=MappingProxyType({seed:tuple(
            (tuple(ScheduledTask(permits[r['task']],r['episode_seed'],r['fast_seed']) for r in row['tasks']),tuple(row['general']))
            for row in actual['train'][str(seed)]) for seed in SEEDS})
        self.identity=_digest(dict(partition=PARTITION,general=GENERAL,schedule=SCHEDULE,tokenizer=tokenizer_identity))

    def batch(self,objective,seed,update):
        if objective not in ('static','adapted') or type(seed) is not int or seed not in SEEDS or type(update) is not int or not 0<=update<64:
            raise ValueError('one original objective/seed/completed cursor required')
        tasks,ids=self._courses[seed][update]
        if len(tasks)!=8 or len(ids)!=8 or len({x.permit.task_id for x in tasks})!=8 or len(set(ids))!=8:
            raise ValueError('eight original tasks and eight general conversations required')
        counts={}
        for x in tasks:counts[x.permit.floor]=counts.get(x.permit.floor,0)+1
        if len(counts)!=4 or set(counts.values())!={2} or {int(f[9:])//100 for f in counts}!={0,2,3,4}:
            raise ValueError('four original room groups with two tasks each required')
        return TrainingBatch(objective,seed,update,tasks,ids,tuple(self.general[k] for k in ids),SCHEDULE)

    def validate_live_batch(self,batch,core,episodes,sessions,general):
        if type(batch) is not TrainingBatch or batch!=self.batch(batch.objective,batch.seed,batch.update):
            raise ValueError('unchanged scheduled batch required')
        if (type(core) is not OutcomeTrajectoryCore or core.config.objective!=batch.objective or core.steps!=batch.update
                or core.config.max_steps!=64 or core.horizon!=64 or core._batch is not None
                or type(episodes) not in (tuple,list) or type(sessions) not in (tuple,list)
                or len(episodes)!=8 or len(sessions)!=8 or tuple(general)!=batch.general):
            raise ValueError('exact complete outer batch and fresh matching slow cursor required')
        for row,episode,session in zip(batch.tasks,episodes,sessions):
            if (type(episode) is not ReplayEpisode or episode.permit!=row.permit or episode.seed!=row.environment_seed
                    or type(session) is not OutcomeSession or session.model is not core.model
                    or session.step or session.fast.steps or session.receipts or episode.checkpoint().steps):
                raise ValueError('actual task/episode course differs from registered slice')
            expected=torch.Generator(device='cpu').manual_seed(row.fast_seed).get_state()
            if (not torch.equal(session.fast.initial_rng,expected) or not torch.equal(session.fast.rng_state,expected)
                    or not torch.equal(session.fast.a,core.model.rho.a) or not torch.equal(session.fast.b,core.model.rho.b)
                    or torch.count_nonzero(session.fast.momentum_a) or torch.count_nonzero(session.fast.momentum_b)):
                raise ValueError('fast seed or original current rho initialization differs')
        return True
