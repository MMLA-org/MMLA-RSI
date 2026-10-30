"""Common admitted role setup and physical-work accounting; no launch on import."""
from pathlib import Path
import json
import sys
import time
from production_common import read,new,atomic,verify_source,EventLog
from proposal_views import attest_role,verify_actor_collection,COLLECTION

def begin(role_name, *, work_directory=None):
    spec=read('/orbit/role.json');source=spec['source_identity'];out=Path('/orbit/work')
    if work_directory is not None:
        requested=Path(work_directory)
        if (role_name not in ('qualification','proposal_reward','actor') or requested.parent!=out/'scalar-workers'
                or requested.name not in tuple('worker-'+str(i) for i in range(4))):
            raise ValueError('only four private worker directories of the admitted role')
        requested.mkdir(exist_ok=False);out=requested
    proof=attest_role(role_name,expected_source=source)
    code=Path('/orbit/code');verify_source(code,source);science=read(code/'scientific-contract.json')
    if (role_name not in ('proposal_reward','actor','qualification') or science['route']!='process-credit-joint-edit-rtma' or science['seeds']!=[17,29,43]
            or science['proposal_training']['effective_updates_per_reader']!=28):
        raise ValueError('exact task-outcome proposal training contract required')
    from importlib.metadata import version
    if {name:version(name) for name in science['model_library_versions']}!=science['model_library_versions']:
        raise ValueError('bound model library versions required')
    new(out/'namespace.json',proof)
    new(out/'model-view.json',verify_actor_collection('/orbit/collection'))
    if any(n=='mmla_pdsa' or n.startswith('mmla_pdsa.') for n in sys.modules):
        raise RuntimeError('fresh original model definitions required')
    sys.path.insert(0,'/orbit/collection/code')
    import torch
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    from deterministic_execution import establish
    new(out/'numerical-execution.json',establish(source))
    from mmla_pdsa.full.outcome_progress import configure
    configure(out,'task-outcome-proposal-'+role_name)
    return source,code,out,science,spec

def original_pairs(code,science,on_event):
    from mmla_pdsa.full.outcome_original import load_original
    from corrected_restore import restore_pair
    original=load_original('/orbit/collection/original-public','/orbit/collection/original-internal',device='cuda:0')
    pairs={}
    for seed in (17,29,43):
        pair=restore_pair(original,'/orbit/collection','/orbit/corrected',key='static-'+str(seed),
            manifest_path=code/'corrected-readers-r1.json',manifest_sha256=science['corrected_readers_sha256'],
            verified_collection_identity=COLLECTION)
        pairs[seed]=pair
        on_event(dict(kind='ORIGINAL_PAIR_RESTORED',seed=seed,restoration=pair['restoration'],inheritance=pair['inheritance']))
    from initial_pairs import restore_initial_pairs
    binding=read(code/'input-binding-r1.json',expected=science['initial_pairs_binding_sha256'])
    pairs=restore_initial_pairs(pairs,binding,directory='/orbit/initial',
        on_restored=lambda value:on_event(dict(kind='PROCESS_CREDIT_INITIAL_PAIR_RESTORED',**value)))
    return original,pairs

class WorkLog:
    def __init__(self,out,phase,limits):
        self.out=Path(out);self.phase=phase;self.limits=limits
        self.events=EventLog(self.out/'events');self.sequence=0;self.last=0.;self.counts={};self.totals={}
    def event(self,value):
        self.sequence+=1
        self.events.write(json.dumps(dict(index=self.sequence,phase=self.phase,event=value),sort_keys=True,allow_nan=False)+'\n')
        now=time.monotonic()
        if now-self.last>=5:
            self.events.flush();atomic(self.out/'progress.json',dict(counter=self.sequence,phase=self.phase,monotonic=now))
            self.last=now
    def work(self,value):
        kind=value['kind'];self.counts[kind]=self.counts.get(kind,0)+1
        for name in ('input_tokens','target_tokens','valid_tokens','padded_positions','tokens','positions','examples','count','updates','candidates'):
            v=value.get(name)
            if type(v) is int:
                if v<0:raise ValueError('negative physical work')
                key=kind+'.'+name;self.totals[key]=self.totals.get(key,0)+v
        self.event(dict(kind='WORK',record=value))
        for key,bound in self.limits.items():
            current=self.totals.get(key,0) if '.' in key else self.counts.get(key,0)
            if current>bound:raise RuntimeError('finite physical work limit exceeded: '+key)
    def close(self):
        self.events.close()

def finish_role(out,*,role,source,spec,result,work):
    import torch
    from mmla_pdsa.full.outcome_progress import finish
    from role_package import assert_frozen_runtime
    from deterministic_execution import require
    require();assert_frozen_runtime('/orbit/collection');torch.cuda.synchronize()
    work.event(dict(kind='ROLE_COMPUTATION_CLOSED',role=role));finish();work.close()
    new(out/'complete.json',dict(schema='orbit-proposal-native-role-v1',source_identity=source,
        job_sha256=spec['job_sha256'],role=role,result=result,counts=work.counts,totals=work.totals,
        status='COMPUTATION_COMPLETE_PENDING_HOST_EXIT',GPU_synchronized=True,
        accounting_complete=False,automatic_retry=False,publication='HOLD'))

def failed(out,role,error):
    new(Path(out)/'INCOMPLETE.json',dict(status='INCOMPLETE',role=role,error=repr(error),
        notes=getattr(error,'__notes__',[]),automatic_retry=False,publication='HOLD'))
