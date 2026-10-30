"""Six complete finite training courses, used only by the production phase entry.

This is not an executable authorization/launch surface. Original loading,
native runtime qualification and resource supervision belong to that entry.
Every step has actual episodes, exact paired course rows and a closed checkpoint.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,math,os,random,time
import torch
from .alfworld_episode import EpisodeLimits,ReplayEpisode,_digest
from .alfworld_formatting import ALFWorldFormatter
from .encoding import FrozenCanonicalEncoder
from .observation_memory import ObservationCodec,observation_memory
from .textworld_backend import TextWorldRuntime,TextWorldProcessBackend
from .extensions.evidence import EvidenceMemory,MemoryBounds
from .extensions.rules import Language,parse_rule
from .extensions.updater import UpdateBounds
from .outcome_original import OriginalOrbit,clone_outcome_model,PUBLIC_MANIFEST
from .outcome_inputs import FiniteInputs,PARTITION,GENERAL,SCHEDULE
from .outcome_training import OutcomeUpdateConfig
from .outcome_trajectory import OutcomeTrajectoryCore
from .outcome_session import OutcomeSession
from .outcome_train_batch import train_outer_batch
from .outcome_checkpoint import CheckpointBinding,save_checkpoint,_ledger
from .neural import tensor_digest

CONTRACT='cba214d207e21a1411837deb5fd35eb5aa980ca756112f372d839dd76b2a6e01'
ORIGINAL_TASK_MODULES='36164d9524e2142d3a9be76b883708f8be0f69296c20090d66aa3574cc6933d1'
TRAINING_ORDER=tuple((objective,seed) for seed in (17,29,43) for objective in ('static','adapted'))


class WorkTotals:
    """Small per-kind numeric aggregates, not a synchronous file write per token."""
    def __init__(self):self.values={}
    def __call__(self,event):
        if type(event) is not dict or type(event.get('kind')) is not str:raise ValueError('identified work event required')
        from .outcome_progress import advance
        advance(event['kind'])
        row=self.values.setdefault(event['kind'],dict(events=0,sums={},maxima={}))
        row['events']+=1
        for key,value in event.items():
            if type(value) not in (int,float):continue
            if not math.isfinite(value):raise ValueError('finite work count required')
            # Cumulative counters/PIDs are not additive resource quantities.
            if key=='pid':continue
            if key.startswith('cumulative_'):row['maxima'][key]=max(row['maxima'].get(key,0),value)
            else:row['sums'][key]=row['sums'].get(key,0)+value
    def environment(self,name,count):self(dict(kind='environment_'+name,count=count))
    def snapshot(self):return json.loads(json.dumps(self.values,allow_nan=False))


class TrainingTrace:
    """Bounded append-only actual-work journal; no live tensors or raw archives.

    Completed event bytes are fsynced before their prefix identity advances.
    A killed process leaves earlier arrived actions/commits identifiable, without
    asserting that an outstanding backend call or gradient batch completed.
    """
    def __init__(self,path):
        self.path=Path(path);self.file=self.path.open('xb',buffering=0)
        self.events=0;self.size=0;self.hash=hashlib.sha256();self.failed=False;self.closed=False
        os.fsync(self.file.fileno())
        fd=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    def write(self,event):
        if self.failed or self.closed:raise RuntimeError('closed or incomplete training trace')
        wire=(json.dumps(event,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
        if len(wire)>256*1024 or self.size+len(wire)>1024**3:raise ValueError('finite per-event/course trace capacity')
        try:
            pending=memoryview(wire)
            while pending:
                n=self.file.write(pending)
                if n is None or n<=0:raise OSError('short progress write')
                pending=pending[n:]
            os.fsync(self.file.fileno())
        except BaseException:self.failed=True;raise
        self.events+=1;self.size+=len(wire);self.hash.update(wire)
        from .outcome_progress import advance
        advance(event.get('kind','committed_trace_event'))
    def receipt(self):
        return dict(path=str(self.path),persisted_events=self.events,persisted_bytes=self.size,
            persisted_sha256=self.hash.hexdigest(),journal_complete=not self.failed,
            observed_bytes=self.path.stat().st_size,closed=self.closed)
    def close(self):
        if not self.closed:
            self.file.close();self.closed=True


class SessionFactory:
    def __init__(self,model,original,*,runtime,on_work):
        if type(original) is not OriginalOrbit or type(runtime) is not TextWorldRuntime or not callable(on_work):
            raise ValueError('verified original and bounded actual backend required')
        self.model,self.original,self.runtime,self.on_work=model,original,runtime,on_work
        tid=original.config['tokenizer_identity'];tok=original.tokenizer
        self.formatter=ALFWorldFormatter(tok,tokenizer_identity=tid,max_tokens=4096)
        self.codec=ObservationCodec(tok,model.profile,tokenizer_identity=tid)
        self.encoder=FrozenCanonicalEncoder(model,tok,tokenizer_identity=tid)
        interface=original.config['proposal_interface'];lang=dict(interface['language']);lang['constants']=tuple(lang['constants'])
        self.language=Language(**lang);self.bounds=UpdateBounds(**interface['update_bounds'])
        self.updater=parse_rule(self.language,'rtt',('c2','c1','c4'))
        self.manager=parse_rule(self.language,'rtma',('sub','c1','x0'))
        self.limits=EpisodeLimits(64,65536,16384,2048,1024,False)
        from .outcome_dense_arithmetic import identity
        from .outcome_generation import GENERATION_ID
        self.definition=dict(generation=GENERATION_ID,execution_arithmetic=identity(model),profile=model.profile.identity,formatter=self.formatter.identity,codec=self.codec.identity,
            encoder=self.encoder.identity,language=asdict(self.language),update_bounds=asdict(self.bounds),
            updater=asdict(self.updater),manager=asdict(self.manager),evaluation_bounds=asdict(MemoryBounds(slots=2,read_tokens=512)),
            episode_limits=asdict(self.limits),runtime=asdict(runtime))

    def backend(self,permit,seed,limits):
        return TextWorldProcessBackend(permit,seed,limits,runtime=self.runtime,on_cost=self.on_work)

    def episode(self,permit,seed):
        return ReplayEpisode(permit,seed=seed,limits=self.limits,factory=self.backend,
            on_work=lambda name,count:self.on_work(dict(kind='environment_'+name,count=count)))

    def reader_identity(self):
        return _digest(dict(original=PUBLIC_MANIFEST,substrate=self.model.substrate_identity,
            modules=tensor_digest(self.model.new_state_dict()),interface=self.definition))

    def session(self,*,seed,tenant,reader_identity):
        return OutcomeSession(self.model,formatter=self.formatter,codec=self.codec,
            source=observation_memory(self.codec,self.encoder),
            evaluation=EvidenceMemory(MemoryBounds(slots=2,read_tokens=512),reader_identity=self.formatter.identity,
                count_tokens=self.formatter.evaluation_token_count,
                namespace='outcome-task-evaluation',source_prefix='environment-observation'),
            reader_identity=reader_identity,seed=seed,tenant=tenant,principal='actor',language=self.language,
            updater=self.updater,manager=self.manager,bounds=self.bounds,max_updates=8)

    def training_batch(self,batch):
        episodes=[];sessions=[];reader=self.reader_identity()
        try:
            for row in batch.tasks:
                episode=self.episode(row.permit,row.environment_seed);episodes.append(episode)
                tenant=_digest(dict(task=row.permit.identity,environment_seed=row.environment_seed,fast_seed=row.fast_seed))
                sessions.append(self.session(seed=row.fast_seed,tenant=tenant,reader_identity=reader))
            return episodes,sessions
        except BaseException as original:
            for episode in episodes:
                try:episode.close()
                except BaseException as error:original.add_note('owned episode cleanup: '+repr(error))
            raise


def _write_new(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,allow_nan=False);f.write('\n');f.flush()
        os.fsync(f.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def run_training(original,inputs,*,runtime,source_identity,output,microbatch,read_ledger):
    """No skipping/resuming/reselection: fixed seed-major 2x3x64 course."""
    if (type(original) is not OriginalOrbit or type(inputs) is not FiniteInputs
            or type(runtime) is not TextWorldRuntime or microbatch not in (8,4,2,1) or type(microbatch) is not int
            or not callable(read_ledger) or len(source_identity)!=64):raise ValueError('complete fixed training dependencies required')
    int(source_identity,16)
    original_modules=tensor_digest(original.task.new_state_dict())
    if (original_modules!=ORIGINAL_TASK_MODULES
            or original.config['children']['task']['modules']!=original_modules
            or original.provenance.get('original_task_modules')!=original_modules
            or original.provenance.get('public_manifest')!=PUBLIC_MANIFEST):
        raise ValueError('all courses must originate from the verified pristine task, never proposal/preflight state')
    out=Path(output);out.mkdir(exist_ok=False);completed=[];active=None;trace=None;cost=None;core=None
    try:
        for objective,seed in TRAINING_ORDER:
            active=dict(objective=objective,seed=seed,next_batch=0,last_closed_checkpoint=None)
            if tensor_digest(original.task.new_state_dict())!=original_modules:raise ValueError('pristine original changed between courses')
            model,migration=clone_outcome_model(original.task)
            torch.manual_seed(seed);random.seed(seed)
            if torch.cuda.is_initialized():torch.cuda.manual_seed_all(seed)
            core=OutcomeTrajectoryCore(model,OutcomeUpdateConfig(CONTRACT,objective,1e-4,64,1.,1.),horizon=64,support_weight=.25)
            cost=WorkTotals();factory=SessionFactory(model,original,runtime=runtime,on_work=cost)
            binding=CheckpointBinding(CONTRACT,SCHEDULE,PARTITION,GENERAL,PUBLIC_MANIFEST,
                original.config['substrate']['theta0'],runtime.dependency_manifest_sha256,source_identity,seed,objective)
            modeldir=out/f'{objective}-{seed}';modeldir.mkdir()
            trace=TrainingTrace(modeldir/'actual-work.jsonl')
            _write_new(modeldir/'course.json',dict(binding=asdict(binding),migration=migration,interface=factory.definition,
                microbatch=microbatch,initial_modules=original_modules,optimizer='fresh AdamW, no inherited moments'))
            last=None
            for update in range(64):
                active['next_batch']=update
                trace.write(dict(kind='BATCH_START',update=update,objective=objective,seed=seed,cumulative_work=cost.snapshot()))
                batch=inputs.batch(objective,seed,update);episodes,sessions=factory.training_batch(batch)
                started=time.monotonic()
                try:
                    inputs.validate_live_batch(batch,core,episodes,sessions,batch.general)
                    result=train_outer_batch(core,episodes,sessions,batch.general,teacher=original.teacher,on_work=cost,microbatch=microbatch,
                        on_progress=lambda event:trace.write(dict(update=update,event=event)))
                except BaseException as error:
                    for episode in episodes:
                        try:episode.close()
                        except BaseException as cleanup:error.add_note('outer cleanup: '+repr(cleanup))
                    raise
                model.module_status['rho']=f'OUTCOME_{objective.upper()}_{core.steps}_SLOW_UPDATES'
                model.module_status['hybrid']=f'OUTCOME_SHARED_READER_{core.steps}_SLOW_UPDATES'
                # Historical risk heads remain byte-preserved but carry no new qualification.
                model.module_status['joint_risk']='HISTORICAL_HEAD_NOT_QUALIFIED_FOR_OUTCOME'
                model.module_status['risk']='HISTORICAL_HEAD_NOT_QUALIFIED_FOR_OUTCOME'
                ledger=read_ledger();_ledger(ledger)
                last=save_checkpoint(core,modeldir/f'step-{core.steps:04d}',binding=binding,ledger=ledger,active_episodes=0)
                active['last_closed_checkpoint']=last
                trace.write(dict(kind='CHECKPOINT_CLOSED',completed_steps=core.steps,checkpoint=last,cumulative_work=cost.snapshot()))
                _write_new(modeldir/f'update-{core.steps:04d}.json',dict(result=result,checkpoint=last,
                    complete_task_wall_seconds=time.monotonic()-started,cumulative_work=cost.snapshot(),ledger=ledger))
            record=dict(objective=objective,seed=seed,updates=core.steps,final_checkpoint=last,
                final_modules=tensor_digest(model.new_state_dict()),work=cost.snapshot())
            trace.close();record['trace']=trace.receipt()
            _write_new(modeldir/'complete.json',record);completed.append(record)
            trace=None;active=None
            del factory,model,episodes,sessions
            core=None;cost=None
            if torch.cuda.is_initialized():torch.cuda.empty_cache()
        if len(completed)!=6:raise RuntimeError('all paired courses must complete')
        result=dict(status='SIX_COURSES_COMPLETE',order=TRAINING_ORDER,models=completed,actual_outer_updates=384)
        _write_new(out/'complete.json',result);return result
    except BaseException as error:
        if trace is not None:
            try:trace.close()
            except BaseException as cleanup:error.add_note('trace close failure: '+repr(cleanup))
        try:_write_new(out/'incomplete.json',dict(status='INCOMPLETE',error=repr(error),completed_courses=completed,
            active_course=active,actual_core_steps=None if core is None else core.steps,
            cumulative_active_work=None if cost is None else cost.snapshot(),
            actual_work_trace=None if trace is None else trace.receipt(),
            outer_batch_open=None if core is None else core._batch is not None,
            continuation_authorized=False))
        except BaseException as reporting:error.add_note('incomplete receipt failure: '+repr(reporting))
        raise
