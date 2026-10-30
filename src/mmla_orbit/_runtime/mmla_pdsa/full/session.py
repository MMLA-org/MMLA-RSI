"""Quiescent task recovery against an already restored, exact model package.

Model/outer-optimizer bundles and live sessions have different ownership. First
load the actual trained model bundle, then import this session into new objects.
This module never replaces trained rho with its initialization. Epoch rebinding
invalidates old candidate/selector/Phi handles; historical receipts remain intact.
"""
from dataclasses import asdict
import copy,json,math,os,random,shutil,tempfile
from pathlib import Path
import torch
from .profile import digest,wire
from .neural import FastState,tensor_digest
from .checkpoint import _fast_validate,_sha
from .runtime import Problem
from .frames import FrameDriver
from .episode_adapter import EngineeringTask
from .encoding import FrozenCanonicalEncoder
from .qualification import FrozenSemanticGate
from .production_memory import QualifiedCanonicalEncoder,production_memory
from .typed_reads import EngineeringViewReader
from .memory import Row


def model_identity(model):
    return digest(dict(profile=model.profile.identity,substrate=model.substrate_identity,
        modules=tensor_digest(model.new_state_dict()),status=model.module_status))


def _contract(task,updater,selector_identity):
    if type(task) is not EngineeringTask:raise ValueError('actual bound engineering task required')
    task.dataset._validate_view(task.view)
    if (type(updater) is not dict or set(updater)!={'learning_rate','momentum','clip'}
            or any(type(x) not in (int,float) or not math.isfinite(x) for x in updater.values())
            or not 0<updater['learning_rate']<=1 or not 0<=updater['momentum']<1 or not 0<updater['clip']<1e6):
        raise ValueError('fixed bounded live RTT updater required')
    if type(selector_identity) is not str or len(selector_identity)!=64:raise ValueError('fixed selector definition required')
    int(selector_identity,16)
    return dict(task=task.identity,source=task.view.identity,dataset=task.dataset.identity,
                timing=task.timing,updater=dict(updater),selector=selector_identity)


def _phase(driver,task,updater):
    p=driver.problem;seg=p.segment;total=len(task.row['observations']['segments'])
    driver._quiescent()
    if (driver._issued is not None or seg._events is not None or seg._cursor!=0 or seg._overlap
            or p.failed or p.closed or p.problem_id!=task.identity or p.tenant!='engineering' or p.principal!='reader'):
        raise ValueError('active complete quiescent task cut required')
    frames=[r for r in p.receipts if r.get('phase')=='FRAME']
    feedback=[r for r in p.receipts if r.get('phase')!='FRAME']
    if (len(frames)!=seg.number or len(frames)>total or len(feedback)!=p.attempts
            or p.attempts not in (0,1) or driver.frame_receipts!=[{k:v for k,v in x.items() if k!='phase'} for x in frames]):
        raise ValueError('actual source cursor/frame receipts/feedback disagree')
    position=0;last_world=0
    for i,receipt in enumerate(frames):
        commands=task.commands(i)
        if (receipt.get('status')!='COMPLETED' or receipt.get('source_frame_number')!=i+1
                or receipt.get('source_receipts')!=[c.envelope.source_receipt for c in commands]
                or len(receipt.get('events',()))!=len(commands)):
            raise ValueError('saved frame does not bind completed source sequence')
        tokens=driver.tokenizer('\n'.join(c.text for c in commands),add_special_tokens=False)['input_ids']
        position+=len(tokens);last_world=commands[-1].envelope.world_clock
    if seg.position!=position or seg.exposed_after!=position:
        raise ValueError('saved source cursor/token exposure disagree')
    if frames:
        last=frames[-1]['events'][-1]
        expected=(last['event'],'NULL' if last['action']=='NULL' else 'COMMIT')
        if seg.exposed_root!=p.memory.identity or seg._last_receipt!=expected:raise ValueError('saved exposure/last event disagrees')
    elif seg.exposed_root is not None or seg._last_receipt is not None or p.memory.version!=0 or any(p.memory._rows):
        raise ValueError('initial task cannot contain prior memory or exposure')
    if feedback:
        f=feedback[0]
        if (f.get('status')!='COMPLETED' or f.get('phi')!='COMMITTED' or f.get('updater')!=updater
                or f.get('memory')!='NULL' or f.get('world_clock')!=task.support_clock
                or p.fast.steps!=1 or p.clock!=2):raise ValueError('exact once-only support update required')
        if len(frames)<(1 if task.timing=='early' else total):raise ValueError('feedback preceded its observation frames')
        expected_world=max(last_world,task.support_clock)
        phase='QUERY_READY' if len(frames)==total else 'SOURCE_POST_SUPPORT'
    else:
        if p.fast.steps!=0 or p.clock!=0:raise ValueError('unexplained numerical/feedback state')
        if task.timing=='early' and len(frames)>1:raise ValueError('early support was skipped')
        phase='SUPPORT_READY' if len(frames)==(1 if task.timing=='early' else total) else 'SOURCE_PRE_SUPPORT'
        expected_world=last_world
    allowed_world={expected_world}
    if len(frames)==total:allowed_world.add(4)
    if p.world_clock not in allowed_world:raise ValueError('source and world clock mismatch')
    if set(p.cost)!={'forwards','backward_updates','generated_tokens','feedback_tokens','memory_reads','memory_commits','wall_seconds'}:
        raise ValueError('complete cumulative runtime cost required')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in p.cost.values()):raise ValueError('invalid accumulated runtime cost')
    return dict(next_source_frame=len(frames),phase=phase)


def _history(driver):
    p=driver.problem;out={}
    if len(driver.rollback)>p.model.profile.memory_slots:raise ValueError('bounded resident parent history')
    for object_id,rows in driver.rollback.items():
        if len(rows)!=1:raise ValueError('one immediate parent per object')
        prior=rows[0];p.memory._validate(prior)
        resident=[r for r in p.memory._rows if r is not None and r.fields['object_id']==object_id
                  and r.fields['incarnation']==prior.fields['incarnation']]
        if len(resident)!=1:raise ValueError('parent no longer belongs to a resident incarnation')
        current=resident[0];f,g=current.fields,prior.fields
        if (current.status=='purged' or g['object_id']!=object_id or f['parent']!=prior.receipt
                or f['version']!=g['version']+1 or f['object_version']!=g['object_version']+1):
            raise ValueError('only actual immediate committed parent may be restored')
        out[object_id]=dict(status=prior.status,**{k:getattr(prior,k).hex() for k in ('canonical','key','value','metadata')})
    return out


def export_session(driver,task,*,updater,selector_identity):
    contract=_contract(task,updater,selector_identity);cursor=_phase(driver,task,updater)
    p=driver.problem;m=p.model;wrapper=p.memory.encoder
    if (type(wrapper) is not QualifiedCanonicalEncoder or type(p.view_reader) is not EngineeringViewReader
            or type(wrapper.encoder) is not FrozenCanonicalEncoder or type(wrapper.gate) is not FrozenSemanticGate):
        raise ValueError('production frozen encoder/gate/typed reader required')
    wrapper.validate_export();_fast_validate(p.fast.__dict__,m,m.new_state_dict())
    fields=('problem_id','tenant','principal','world_clock','clock','attempts','receipts','cost','closed','failed','_reader','last_read_identity')
    payload=dict(schema='mmla-quiescent-session-v1',model=model_identity(m),contract=contract,cursor=cursor,
        tokenizer=driver.tokenizer_identity,profile=m.profile.identity,encoder=wrapper.encoder.identity,gate=wrapper.gate.export_state(),
        memory=p.memory.snapshot(),memory_identity=p.memory.identity,phi_identity=p.fast.identity,
        fast={k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in p.fast.__dict__.items()},
        problem={k:copy.deepcopy(getattr(p,k)) for k in fields},
        segment={k:copy.deepcopy(getattr(p.segment,k)) for k in ('position','number','exposed_root','exposed_after','_last_receipt')},
        history=_history(driver),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
        cuda_rng=None if m.rho.a.device.type!='cuda' else torch.cuda.get_rng_state(m.rho.a.device))
    return payload


def import_session(model,tokenizer,task,payload,*,tokenizer_identity,updater,selector_identity):
    """Build new objects; no mutation of the live model or global RNG on failure.

    Owner applies returned RNG only after publishing the validated object graph.
    Physical cost of gate revalidation is additionally charged by the run meter.
    """
    if (payload['schema']!='mmla-quiescent-session-v1' or payload['model']!=model_identity(model)
            or payload['profile']!=model.profile.identity or payload['tokenizer']!=tokenizer_identity
            or payload['contract']!=_contract(task,updater,selector_identity)):
        raise ValueError('session requires its exact trained model, source and runtime definitions')
    problem_keys={'problem_id','tenant','principal','world_clock','clock','attempts','receipts','cost','closed','failed','_reader','last_read_identity'}
    segment_keys={'position','number','exposed_root','exposed_after','_last_receipt'}
    if set(payload['problem'])!=problem_keys or set(payload['segment'])!=segment_keys:
        raise ValueError('complete exact runtime and segment fields required')
    for source,keys in ((payload['problem'],('world_clock','clock','attempts')),(payload['segment'],('position','number','exposed_after'))):
        if any(type(source[k]) is not int or not 0<=source[k]<2**63-1 for k in keys):raise ValueError('bounded exact recovery counters')
    if type(payload['problem']['closed']) is not bool or type(payload['problem']['failed']) is not bool:raise ValueError('exact terminal state flags')
    _fast_validate(payload['fast'],model,model.new_state_dict())
    if FastState(**payload['fast']).identity!=payload['phi_identity']:raise ValueError('original fast state receipt mismatch')
    if payload['fast']['epoch']>=2**63-2:raise ValueError('fast recovery epoch exhausted')
    torch.Generator(device='cpu').set_state(payload['torch_rng']);random.Random().setstate(payload['python_rng'])
    if any(module.training for module in model.mixer.modules()):
        raise ValueError("shared frozen mixer must already be in eval mode")
    encoder=FrozenCanonicalEncoder(model,tokenizer,tokenizer_identity=tokenizer_identity)
    if encoder.identity!=payload['encoder']:raise ValueError('original frozen canonical encoder cannot be reconstructed')
    gate=FrozenSemanticGate.from_state(model,encoder,tokenizer,tokenizer_identity=tokenizer_identity,state=payload['gate'])
    memory=production_memory(model.profile,encoder,gate);memory.restore(payload['memory'],rollback=True)
    # Verify old content identity independently of the intentional new epoch.
    old_content=dict(profile=model.profile.identity,encoder=memory.encoder.identity,
        epoch=payload['memory']['epoch'],version=memory.version,
        rows=[None if r is None else r.receipt for r in memory._rows])
    if payload['memory'].get('recovery_id'):old_content['recovery_id']=payload['memory']['recovery_id']
    old=digest(old_content)
    if old!=payload['memory_identity']:raise ValueError('original memory receipt mismatch')
    reader=EngineeringViewReader(memory.encoder,gate)
    p=Problem(model,memory,problem_id=task.identity,tenant='engineering',principal='reader',seed=0,view_reader=reader)
    actual_reader=p.reader_identity()
    if payload['problem']['_reader']!=actual_reader:raise ValueError('saved task reader changed')
    for key,value in payload['problem'].items():
        if key not in ('problem_id','tenant','principal','world_clock','clock','attempts','receipts','cost','closed','failed','_reader','last_read_identity'):
            raise ValueError('unexpected runtime field')
        setattr(p,key,copy.deepcopy(value))
    data={k:v.to(model.rho.a.device).detach().clone() if torch.is_tensor(v) and k not in ('rng_state','initial_rng') else copy.deepcopy(v)
          for k,v in payload['fast'].items()}
    data['a']=data['a'].detach().clone().requires_grad_();data['b']=data['b'].detach().clone().requires_grad_()
    data['epoch']+=1;data['rho_identity']=model.rho.identity
    data['recovery_id']=memory.recovery_id;p.fast=FastState(**data)
    for key,value in payload['segment'].items():
        if key not in ('position','number','exposed_root','exposed_after','_last_receipt'):raise ValueError('unexpected segment field')
        setattr(p.segment,key,copy.deepcopy(value))
    if p.segment.exposed_root is not None:
        if p.segment.exposed_root!=old:raise ValueError('old exposure not bound to saved root')
        p.segment.exposed_root=memory.identity
    # Historical read receipt is retained separately, never used as a new read.
    old_read=p.last_read_identity;p.last_read_identity=None
    driver=FrameDriver(p,tokenizer,tokenizer_identity=tokenizer_identity)
    driver.frame_receipts=[{k:v for k,v in r.items() if k!='phase'} for r in p.receipts if r.get('phase')=='FRAME']
    driver.rollback={obj:[Row(row['status'],*(bytes.fromhex(row[k]) for k in ('canonical','key','value','metadata')))] for obj,row in payload['history'].items()}
    _history(driver)
    if _phase(driver,task,updater)!=payload['cursor']:raise ValueError('restored phase/cursor mismatch')
    # Rebinding changes handles, never the adapted numerical state.
    for key in ('a','b','momentum_a','momentum_b','rng_state','initial_rng'):
        if not torch.equal(getattr(p.fast,key).cpu(),payload['fast'][key].cpu()):raise ValueError('fast numerical restore mismatch')
    receipt=dict(kind='QUIESCENT_RECOVERY_REBOUND',old_memory=old,new_memory=memory.identity,
        old_phi=payload['phi_identity'],new_phi=p.fast.identity,old_read=old_read,recovery_id=memory.recovery_id,
        cached_selectors_candidates_and_exact_phi_calibration='INVALIDATED_RECOMPUTE',cursor=payload['cursor'])
    return dict(driver=driver,receipt=receipt,torch_rng=payload['torch_rng'],python_rng=payload['python_rng'],cuda_rng=payload['cuda_rng'])


def save_session(driver,task,destination,*,updater,selector_identity):
    payload=export_session(driver,task,updater=updater,selector_identity=selector_identity)
    dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError('session versions are immutable')
    tmp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        state=tmp/'session.pt';torch.save(payload,state)
        if state.stat().st_size>64*1024**2:raise ValueError('bounded full session artifact')
        manifest=dict(schema=1,model=payload['model'],contract=payload['contract'],cursor=payload['cursor'],
            state=dict(name='session.pt',bytes=state.stat().st_size,sha256=_sha(state)))
        (tmp/'manifest.json').write_bytes(wire(manifest)+b'\n')
        for path in (state,tmp/'manifest.json'):
            with path.open('rb') as f:os.fsync(f.fileno())
        fd=os.open(tmp,os.O_RDONLY);os.fsync(fd);os.close(fd)
        if dest.exists():raise FileExistsError(dest)
        os.rename(tmp,dest);fd=os.open(dest.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
        return manifest
    except Exception:
        shutil.rmtree(tmp,ignore_errors=True);raise


def load_session(model,tokenizer,task,directory,*,tokenizer_identity,updater,selector_identity):
    path=Path(directory);manifest=json.loads((path/'manifest.json').read_bytes());row=manifest['state']
    if manifest['schema']!=1 or row['name']!='session.pt' or not 0<row['bytes']<=64*1024**2:raise ValueError('bounded session manifest')
    state=path/row['name']
    if state.stat().st_size!=row['bytes'] or _sha(state)!=row['sha256']:raise ValueError('session bytes changed')
    payload=torch.load(state,map_location='cpu',weights_only=True)
    if any(manifest[k]!=payload[k] for k in ('model','contract','cursor')):raise ValueError('session header mismatch')
    return import_session(model,tokenizer,task,payload,tokenizer_identity=tokenizer_identity,updater=updater,selector_identity=selector_identity)
