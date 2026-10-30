"""Restore the six completed courses for evaluation; never resume training.

Checkpoint resources must be a verified prefix of the same service's durable
ledger. Every reader is cloned from the original task and restored at step 64.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,random
import torch
from .outcome_checkpoint import CheckpointBinding,load_checkpoint,_ledger
from .outcome_train_stage import CONTRACT,TRAINING_ORDER,ORIGINAL_TASK_MODULES
from .outcome_inputs import SCHEDULE,PARTITION,GENERAL
from .outcome_original import clone_outcome_model,PUBLIC_MANIFEST
from .outcome_training import OutcomeUpdateConfig
from .outcome_trajectory import OutcomeTrajectoryCore
from .extensions.rsi_binding import rebind
from .neural import tensor_digest

FIELDS=('run_id','started_unix','deadline_unix','wall_seconds','CPU_seconds','GPU_seconds','events_count','events_sha256')
COSTS=('wall_seconds','CPU_seconds','GPU_seconds')


def checkpoint_fields(row):
    result={k:row[k] for k in FIELDS};_ledger(result);return result


class LedgerEvidence:
    """An immutable verified current prefix, including actual old checkpoint cuts.

    The external supervisor alone appends and publishes resource-latest.json.
    Reading a prefix during a later append is safe: exact count/hash terminate
    this read. There is no retry, new origin or discounted continuation ledger.
    """
    def __init__(self,run_root,*,run_id,started_unix,requested):
        root=Path(run_root);latest=json.loads((root/'resource-latest.json').read_bytes())
        current=checkpoint_fields(latest)
        if current['deadline_unix'] is None:
            from .outcome_workload_resources import POLICY
            admission=json.loads((root.parent/'admission.json').read_bytes())
            if (latest.get('time_policy')!=POLICY or admission.get('time_policy')!=POLICY
                    or admission.get('run_id')!=run_id or admission.get('started_unix')!=started_unix):
                raise ValueError('explicit admitted workload policy required for no deadline')
        elif current['deadline_unix']!=started_unix+172800.:raise ValueError('original finite deadline')
        if current['run_id']!=run_id or current['started_unix']!=started_unix:raise ValueError('original service identity/origin')
        if latest.get('failed_segments') or latest.get('hard_limit_exceeded'):raise ValueError('failed/budget-exceeded service cannot load evaluation readers')
        wanted={}
        for row in requested:
            _ledger(row);n=row['events_count']
            if not 0<n<=current['events_count']<=400000:raise ValueError('nonempty bounded checkpoint event prefix')
            if n in wanted and wanted[n]!=row:raise ValueError('inconsistent same-prefix checkpoint costs')
            if any(row[k]!=current[k] for k in ('run_id','started_unix','deadline_unix')):raise ValueError('checkpoint original service differs')
            if any(row[k]>current[k] for k in COSTS):raise ValueError('checkpoint discounts later costs')
            wanted[n]=row
        digest=hashlib.sha256();previous=None;matched=[];total=0
        with (root/'resource-events.jsonl').open('rb') as f:
            for i in range(current['events_count']):
                line=f.readline(32769);total+=len(line)
                if not line.endswith(b'\n') or len(line)>32768 or total>512*1024**2:raise ValueError('bounded complete durable resource event')
                event=json.loads(line);row=checkpoint_fields(event['resources'])
                if (row['events_count']!=i or row['events_sha256']!=digest.hexdigest()
                        or any(row[k]!=current[k] for k in ('run_id','started_unix','deadline_unix'))):raise ValueError('resource hash chain/origin differs')
                if previous is not None and any(row[k]<previous[k] for k in COSTS):raise ValueError('resource costs moved backwards')
                digest.update(line)
                bound=dict(row,events_count=i+1,events_sha256=digest.hexdigest())
                if i+1 in wanted:
                    if bound!=wanted[i+1]:raise ValueError('checkpoint not equal to actual durable resource prefix')
                    matched.append(bound)
                previous=row
        if digest.hexdigest()!=current['events_sha256'] or previous is None or any(previous[k]!=current[k] for k in COSTS):
            raise ValueError('published latest resource receipt differs from journal')
        self.current=current;self.matched=tuple(matched)
    def reconcile(self,row):
        _ledger(row)
        return row in self.matched


def course_receipts(training_root):
    root=Path(training_root);complete=json.loads((root/'complete.json').read_bytes())
    if (complete.get('status')!='SIX_COURSES_COMPLETE' or complete.get('actual_outer_updates')!=384
            or tuple(map(tuple,complete['order']))!=TRAINING_ORDER
            or [(x['objective'],x['seed']) for x in complete['models']]!=list(TRAINING_ORDER)):
        raise ValueError('all six original finite courses must have completed')
    rows={}
    for row in complete['models']:
        objective,seed=row['objective'],row['seed'];course=root/f'{objective}-{seed}'
        if json.loads((course/'complete.json').read_bytes())!=row or row['updates']!=64 or row['final_checkpoint']['next_batch']!=64:
            raise ValueError('full closed 64-step course required')
        description=json.loads((course/'course.json').read_bytes());identity=row['final_checkpoint']['identity']
        from .artifacts import load_artifact
        from .outcome_checkpoint import KIND
        payload=load_artifact(course/'step-0064',kind=KIND,identity=identity)
        if payload['module_identity']!=row['final_modules'] or payload['next_batch']!=64 or payload['binding']!=description['binding']:
            raise ValueError('actual final checkpoint/course agreement')
        rows[(objective,seed)]=dict(record=row,course=description,payload=payload,directory=course/'step-0064')
    return rows


def restore_readers(original,training_root,*,runtime,source_identity,run_root,run_id,started_unix):
    """All six final readers, no seed selection or optimizer continuation."""
    if tensor_digest(original.task.new_state_dict())!=ORIGINAL_TASK_MODULES:raise ValueError('pristine original task required')
    rows=course_receipts(training_root)
    ledger=LedgerEvidence(run_root,run_id=run_id,started_unix=started_unix,requested=[x['payload']['ledger'] for x in rows.values()])
    readers={};receipts={};python_rng=random.getstate()
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            for objective,seed in TRAINING_ORDER:
                row=rows[(objective,seed)]
                binding=CheckpointBinding(CONTRACT,SCHEDULE,PARTITION,GENERAL,PUBLIC_MANIFEST,
                    original.config['substrate']['theta0'],runtime.dependency_manifest_sha256,source_identity,seed,objective)
                if asdict(binding)!=row['course']['binding']:raise ValueError('evaluation binding changed from training')
                model,_=clone_outcome_model(original.task)
                core=OutcomeTrajectoryCore(model,OutcomeUpdateConfig(CONTRACT,objective,1e-4,64,1.,1.),horizon=64,support_weight=.25)
                result=load_checkpoint(core,row['directory'],identity=row['record']['final_checkpoint']['identity'],
                    binding=binding,reconcile_ledger=ledger.reconcile,active_episodes=0)
                if core.steps!=64 or tensor_digest(model.new_state_dict())!=row['record']['final_modules']:
                    raise ValueError('actual final reader restore failed')
                model.eval().requires_grad_(False);rebind(model)
                readers[(objective,seed)]=model;receipts[(objective,seed)]=result
                del core # AdamW moments were checked/restored, then released; no resume.
    finally:random.setstate(python_rng)
    return readers,receipts,ledger.current
