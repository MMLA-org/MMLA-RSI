"""Complete quiet token-stream recovery, separate from the old frame format.

A quiet cut may contain a partial source command or a stopped prefix not yet
processed. In-flight compiler/selector/commit work is terminal, not replayed.
The object-format alone does not authorize replay of an old quiet file after
a later operation. StreamCheckpointRun fences production operations using the
existing Journal and restores only its latest QUIET cut.
Import builds new state, invalidates old M/Phi handles and leaves global RNG
untouched. Physical encoder/gate reconstruction belongs to the run cost meter.
"""
import copy, json, math, os, random, shutil, tempfile
from pathlib import Path
import torch
from ..checkpoint import _fast_validate, _sha
from ..encoding import FrozenCanonicalEncoder
from ..qualification import FrozenSemanticGate
from ..production_memory import QualifiedCanonicalEncoder, production_memory
from ..typed_reads import EngineeringViewReader
from ..runtime import Problem
from ..memory import Row
from ..neural import FastState
from ..profile import digest, wire
from ..session import model_identity, _history
from ..local_rng import capture_cuda, validate_cuda
from .stream import StreamDriver

FIELDS=('problem_id','tenant','principal','world_clock','clock','attempts','receipts',
        'cost','closed','failed','_reader','last_read_identity')
SCHEMA='mmla-stream-session-v1'
JOINT_SCHEMA='mmla-joint-stream-session-v1'
MAX_BYTES=64*1024**2


def _phase(driver):
    driver._quiet();p=driver.problem;s=p.segment
    state=driver.export_collecting()
    streams=[r for r in p.receipts if r.get('phase')=='STREAM']
    feedback=[r for r in p.receipts if r.get('phase')!='STREAM']
    if (streams!=[dict(phase='STREAM',**r) for r in driver.receipts]
            or len(feedback)!=p.attempts or not 0<=p.attempts<=p.model.profile.max_attempts
            or any(r.get('status')!='COMPLETED' for r in feedback)):
        raise ValueError('complete successful source/feedback history required')
    from .joint import JointStreamDriver
    joint=type(driver) is JointStreamDriver
    updates=computed=0
    joint_events=[e for r in streams for e in r['events'] if e.get('selection',{}).get('mode')=='PDSA_J']
    joint_feedback=[r for r in feedback if r.get('phase')=='JOINT_FEEDBACK']
    if len(joint_events)!=len(joint_feedback):raise ValueError('every J feedback needs one actual event decision')
    if joint and len(joint_feedback)!=len(feedback):raise ValueError('joint mode cannot contain ordinary immediate feedback')
    for r in feedback:
        if r.get('phase')=='JOINT_FEEDBACK':
            prep=r.get('preparation',{});phi=r.get('phi');cost=prep.get('cost',{})
            if (not joint or phi not in ('KEEP','COMMITTED') or r.get('memory') not in ('NULL','WRITE')
                    or r.get('updater')!=driver.updater or prep.get('updater')!=driver.updater
                    or prep.get('status')!='PREPARED' or prep.get('before')!=r.get('phi_before')
                    or prep.get('read_clock')!=r.get('world_clock')
                    or prep.get('decision_clock')!=r.get('decision_clock')
                    or not math.isfinite(prep.get('loss',float('nan')))
                    or cost.get('backward_updates')!=1 or cost.get('forwards')!=1
                    or cost.get('feedback_tokens',0)<=0
                    or r.get('phi_after')!=(prep.get('after') if phi=='COMMITTED' else r.get('phi_before'))):
                raise ValueError('complete prepared-versus-committed J feedback required')
            events=[e for e in joint_events if e['selection'].get('anchor')==r.get('anchor')]
            if (len(events)!=1 or events[0]['selection'].get('pair')!=r.get('pair')
                    or events[0].get('action')!=r['memory']
                    or events[0].get('predecessor')!=r.get('memory_before')
                    or events[0].get('successor')!=r.get('memory_after')):
                raise ValueError('J receipt does not match actual source commit')
            computed+=1;updates+=int(phi=='COMMITTED')
        elif driver.updater is None:
            if r.get('phi')!='NO_UPDATE' or 'updater' in r or r.get('memory')!='NULL':raise ValueError('undeclared numerical update')
        else:
            if r.get('phi')!='COMMITTED' or r.get('updater')!=driver.updater or r.get('memory')!='NULL':
                raise ValueError('saved feedback used another updater')
            updates+=1;computed+=1
    if p.fast.steps!=updates:raise ValueError('unexplained numerical state changes')
    if s.number:
        if s.exposed_root!=p.memory.identity:raise ValueError('published source root differs from saved memory')
    elif (p.memory.version or any(p.memory._rows) or s.exposed_root is not None):
        raise ValueError('initial source cannot contain unrecorded memory')
    expected={'forwards','backward_updates','generated_tokens','feedback_tokens','memory_reads','memory_commits','wall_seconds'}
    if (set(p.cost)!=expected or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in p.cost.values())
            or p.cost['backward_updates']!=computed):
        raise ValueError('complete cumulative runtime cost required')
    return state


def export_stream_session(driver):
    from .joint import JointStreamDriver
    if type(driver) not in (StreamDriver,JointStreamDriver):raise ValueError('actual token-stream driver required')
    state=_phase(driver);p=driver.problem;m=p.model;wrapper=p.memory.encoder
    if (type(wrapper) is not QualifiedCanonicalEncoder or type(wrapper.encoder) is not FrozenCanonicalEncoder
            or type(wrapper.gate) is not FrozenSemanticGate or type(p.view_reader) is not EngineeringViewReader):
        raise ValueError('production encoder, gate and typed reader required')
    wrapper.validate_export();_fast_validate(p.fast.__dict__,m,m.new_state_dict())
    return dict(schema=JOINT_SCHEMA if type(driver) is JointStreamDriver else SCHEMA,model=model_identity(m),contract=driver.contract,profile=m.profile.identity,
        tokenizer=driver.tokenizer_identity,source=driver.source_identity,encoder=wrapper.encoder.identity,
        gate=wrapper.gate.export_state(),memory=p.memory.snapshot(),memory_identity=p.memory.identity,
        phi_identity=p.fast.identity,
        fast={k:v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in p.fast.__dict__.items()},
        problem={k:copy.deepcopy(getattr(p,k)) for k in FIELDS},stream=state,history=_history(driver),
        torch_rng=torch.get_rng_state().clone(),python_rng=random.getstate(),cuda_rng=capture_cuda())


def import_stream_session(model,tokenizer,payload,*,tokenizer_identity,source_identity,
                          boundary,limits,selector_identity,updater=None,joint_selector_identity=None,endpoint=None):
    """Callers supply the original definitions, never infer a new task from a snapshot."""
    from .joint import JointStreamDriver
    if joint_selector_identity is None and endpoint is not None:raise ValueError('joint endpoint supplied to ordinary stream')
    schema=SCHEMA if joint_selector_identity is None else JOINT_SCHEMA
    required={'schema','model','contract','profile','tokenizer','source','encoder','gate','memory',
        'memory_identity','phi_identity','fast','problem','stream','history','torch_rng','python_rng','cuda_rng'}
    if (set(payload)!=required or payload['schema']!=schema or payload['model']!=model_identity(model)
            or payload['profile']!=model.profile.identity or payload['tokenizer']!=tokenizer_identity
            or payload['source']!=source_identity or set(payload['problem'])!=set(FIELDS)):
        raise ValueError('complete exact trained-model/source session required')
    fields=payload['problem']
    for k in ('world_clock','clock','attempts'):
        if type(fields[k]) is not int or not 0<=fields[k]<2**63-1:raise ValueError('bounded exact runtime counter')
    if fields['closed'] is not False or fields['failed'] is not False:
        raise ValueError('only active quiet sessions can resume')
    _fast_validate(payload['fast'],model,model.new_state_dict())
    if FastState(**payload['fast']).identity!=payload['phi_identity'] or payload['fast']['epoch']>=2**63-2:
        raise ValueError('original fast identity/epoch differs')
    torch.Generator(device='cpu').set_state(payload['torch_rng']);random.Random().setstate(payload['python_rng'])
    validate_cuda(payload['cuda_rng'])
    if any(m.training for m in model.mixer.modules()):raise ValueError('shared frozen mixer must already be eval')
    enc=FrozenCanonicalEncoder(model,tokenizer,tokenizer_identity=tokenizer_identity)
    if enc.identity!=payload['encoder']:raise ValueError('original encoder cannot be reconstructed')
    gate=FrozenSemanticGate.from_state(model,enc,tokenizer,tokenizer_identity=tokenizer_identity,state=payload['gate'])
    memory=production_memory(model.profile,enc,gate);memory.restore(payload['memory'],rollback=True)
    content=dict(profile=model.profile.identity,encoder=memory.encoder.identity,epoch=payload['memory']['epoch'],
        version=memory.version,rows=[None if r is None else r.receipt for r in memory._rows])
    if payload['memory'].get('recovery_id'):content['recovery_id']=payload['memory']['recovery_id']
    old=digest(content)
    if old!=payload['memory_identity']:raise ValueError('original memory content identity differs')
    p=Problem(model,memory,problem_id=fields['problem_id'],tenant=fields['tenant'],principal=fields['principal'],
        seed=0,view_reader=EngineeringViewReader(memory.encoder,gate))
    if p._reader!=fields['_reader']:raise ValueError('saved task reader changed')
    for k in FIELDS:setattr(p,k,copy.deepcopy(fields[k]))
    data={k:v.to(model.rho.a.device).detach().clone() if torch.is_tensor(v) and k not in ('rng_state','initial_rng')
          else copy.deepcopy(v) for k,v in payload['fast'].items()}
    data['a']=data['a'].detach().clone().requires_grad_();data['b']=data['b'].detach().clone().requires_grad_()
    data['epoch']+=1;data['rho_identity']=model.rho.identity;data['recovery_id']=memory.recovery_id
    p.fast=FastState(**data)
    cls=StreamDriver if joint_selector_identity is None else JointStreamDriver
    extra={} if joint_selector_identity is None else dict(joint_selector_identity=joint_selector_identity,endpoint=endpoint)
    driver=cls(p,tokenizer,tokenizer_identity=tokenizer_identity,source_identity=source_identity,
                        boundary=boundary,limits=limits,selector_identity=selector_identity,updater=updater,**extra)
    if driver.contract!=payload['contract']:raise ValueError('original stream configuration required')
    state=copy.deepcopy(payload['stream'])
    root=state['clock']['exposed_root']
    if root is not None:
        if root!=old:raise ValueError('old publication root mismatch')
        state['clock']['exposed_root']=memory.identity
    driver.restore_collecting(state)
    driver.rollback={obj:[Row(row['status'],*(bytes.fromhex(row[k]) for k in ('canonical','key','value','metadata')))]
                     for obj,row in payload['history'].items()}
    _history(driver);_phase(driver)
    previous_read=p.last_read_identity;p.last_read_identity=None
    for k in ('a','b','momentum_a','momentum_b','rng_state','initial_rng'):
        if not torch.equal(getattr(p.fast,k).cpu(),payload['fast'][k].cpu()):raise ValueError('numerical/RNG restore mismatch')
    receipt=dict(kind='QUIET_STREAM_RECOVERY_REBOUND',old_memory=old,new_memory=memory.identity,
        old_phi=payload['phi_identity'],new_phi=p.fast.identity,old_read=previous_read,recovery_id=memory.recovery_id,
        next_source_token=p.segment.position+1,next_command=driver._processed_command+1,
        cached_selectors_candidates_and_exact_phi_calibration='INVALIDATED_RECOMPUTE')
    return dict(driver=driver,receipt=receipt,torch_rng=payload['torch_rng'].clone(),
                python_rng=copy.deepcopy(payload['python_rng']),cuda_rng=copy.deepcopy(payload['cuda_rng']))


def save_stream_session(driver,destination):
    payload=export_stream_session(driver);dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError('session versions are immutable')
    tmp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        state=tmp/'session.pt';torch.save(payload,state)
        if state.stat().st_size>MAX_BYTES:raise ValueError('bounded full stream artifact')
        manifest=dict(schema=payload['schema'],model=payload['model'],contract=payload['contract'],
            state=dict(name='session.pt',bytes=state.stat().st_size,sha256=_sha(state)))
        (tmp/'manifest.json').write_bytes(wire(manifest)+b'\n')
        for path in (state,tmp/'manifest.json'):
            with path.open('rb') as f:os.fsync(f.fileno())
        fd=os.open(tmp,os.O_RDONLY);os.fsync(fd);os.close(fd)
        if dest.exists():raise FileExistsError(dest)
        os.rename(tmp,dest);fd=os.open(dest.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
        return manifest
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise


def load_stream_session(model,tokenizer,directory,**definitions):
    path=Path(directory);header=path/'manifest.json'
    if header.stat().st_size>16384:raise ValueError('bounded stream manifest')
    manifest=json.loads(header.read_bytes());row=manifest['state']
    if manifest['schema'] not in (SCHEMA,JOINT_SCHEMA) or row['name']!='session.pt' or not 0<row['bytes']<=MAX_BYTES:
        raise ValueError('bounded exact stream manifest')
    state=path/row['name']
    if state.stat().st_size!=row['bytes'] or _sha(state)!=row['sha256']:raise ValueError('stream artifact bytes changed')
    payload=torch.load(state,map_location='cpu',weights_only=True)
    if any(manifest[k]!=payload[k] for k in ('schema','model','contract')):raise ValueError('stream header mismatch')
    return import_stream_session(model,tokenizer,payload,**definitions)

class StreamCheckpointRun:
    """Existing Journal supplies a durable, single-writer operation fence.

    This is the production entry around a trusted finite operation such as
    feeding one segment and processing its stop, or propose+feedback. Callbacks
    must return at a quiet cut. RUNNING/FAILED latest cuts cannot be resumed;
    archived snapshots remain artifacts, not alternative run-admission points.
    """
    KIND='mmla-stream-run-v1'

    def __init__(self,driver,journal):
        from .journal import Journal
        if type(journal) is not Journal:raise ValueError('bounded local Journal required')
        self.driver,self.journal=driver,journal
        self.step=0;self.cut=None;self.failed=False
        with journal.transaction(2,expected_cut=None):
            self.cut=journal.save(dict(kind=self.KIND,status='RUNNING',step=0,operation='initialize'))
            try:
                state=export_stream_session(driver)
                self.cut=journal.save(dict(kind=self.KIND,status='QUIET',step=0,session=state))
            except BaseException as error:
                self.failed=True
                try:self.cut=journal.save(dict(kind=self.KIND,status='FAILED',step=0,operation='initialize',error=type(error).__name__))
                except BaseException:pass
                raise

    @classmethod
    def restore(cls,model,tokenizer,journal,**definitions):
        cut,state=journal.load_bound()
        if (set(state)!={'kind','status','step','session'} or state['kind']!=cls.KIND
                or state['status']!='QUIET' or type(state['step']) is not int or state['step']<0):
            raise ValueError('only latest successful quiet stream cut may resume')
        # Qualification/encoding on import and validation on export are actual
        # work. The outer meter covers this transaction, not just callbacks.
        with journal.transaction(2,expected_cut=cut):
            running=journal.save(dict(kind=cls.KIND,status='RUNNING',step=state['step'],operation='recovery',prior=cut))
            try:
                result=import_stream_session(model,tokenizer,state['session'],**definitions)
                rebound=export_stream_session(result['driver'])
                published=journal.save(dict(kind=cls.KIND,status='QUIET',step=state['step'],session=rebound))
            except BaseException as error:
                try:journal.save(dict(kind=cls.KIND,status='FAILED',step=state['step'],operation='recovery',prior=cut,error=type(error).__name__))
                except BaseException:pass
                raise
        obj=cls.__new__(cls);obj.driver=result['driver'];obj.journal=journal
        obj.cut=published;obj.step=state['step'];obj.failed=False
        result['run']=obj
        return result

    def execute(self,operation,callback):
        if self.failed:raise RuntimeError('failed stream run cannot execute again')
        if operation not in ('source_segment','feedback_cycle','read_only_query','joint_cycle'):
            raise ValueError('registered finite operation required')
        # Pure runtime admission only; encoder/gate export validation is fenced.
        self.driver._quiet()
        with self.journal.transaction(2,expected_cut=self.cut):
            p=self.driver.problem
            self.cut=self.journal.save(dict(kind=self.KIND,status='RUNNING',step=self.step+1,
                operation=operation,pre_phi=p.fast.identity,pre_memory=p.memory.identity))
            try:
                export_stream_session(self.driver)
                result=callback(self.driver)
                state=export_stream_session(self.driver)
                self.cut=self.journal.save(dict(kind=self.KIND,status='QUIET',step=self.step+1,session=state))
                self.step+=1
                return result
            except BaseException as error:
                self.failed=True;p.failed=True
                failure=dict(kind=self.KIND,status='FAILED',step=self.step+1,operation=operation,
                    error=type(error).__name__,phi=p.fast.identity,memory=p.memory.identity,
                    runtime_cost=copy.deepcopy(p.cost),stream_cost=copy.deepcopy(self.driver.cost))
                # A storage fault may prevent FAILED publication. The already
                # durable RUNNING cut (or orphan publication) still fences replay.
                try:self.cut=self.journal.save(failure)
                except BaseException:pass
                raise
