"""Common-predecessor Phi x M decisions on the next real source event.

Arrived support is sealed at its original read clock. No source event may commit
between arrival and the anchor. Numerical preparation is paid even for KEEP.
This fixed-SGD implementation is distinct from the recursive RSI U interpreter.
"""
from dataclasses import dataclass,asdict
import copy,json,math,time
import torch
from ..profile import digest,wire
from ..runtime import Feedback
from ..neural import tensor_digest
from ..frames import FrameDriver,action_identity,BranchFailure
from ..q_features import action_features
from .stream import StreamDriver

JOINT_SCHEMA=digest(dict(schema='mmla-joint-action-features-v1',
    base='mmla-current-action-features-v1',extra=('commit_phi','tanh_delta_A','tanh_delta_B',
    'tanh_delta_momentum_A','tanh_delta_momentum_B','tanh_support_CE'),
    actions='KEEP x all M, then UPDATE x all M; ordered complete domain',
    access='current event, current state, arrived support only'))

@dataclass(frozen=True)
class ArrivedFeedback:
    feedback:Feedback
    attempt:str
    phi:str
    memory:str
    reader:str
    read_clock:int
    source_position:int
    completed_command:int
    source_receipt:str
    initial_rng:str
    @property
    def identity(self):return digest(asdict(self))

@dataclass(frozen=True)
class JointFeatures:
    matrix:tuple
    pairs:tuple
    binding:dict
    def __post_init__(self):
        n=len(self.pairs)//2
        if (not n or len(self.pairs)!=2*n or len(self.matrix)!=2*n
                or self.pairs[0]!=(0,'NULL') or self.pairs[n]!=(1,'NULL')
                or tuple(x[0] for x in self.pairs)!=(0,)*n+(1,)*n
                or tuple(x[1] for x in self.pairs[:n])!=tuple(x[1] for x in self.pairs[n:])
                or len({x[1] for x in self.pairs[:n]})!=n or not self.matrix[0]
                or any(len(row)!=len(self.matrix[0]) or any(not math.isfinite(v) for v in row) for row in self.matrix)):
            raise ValueError('complete finite KEEP/UPDATE x common M feature matrix required')
    @property
    def identity(self):return digest(asdict(self))
    def tensor(self,device):return torch.tensor(self.matrix,device=device,dtype=torch.float32)

@dataclass(frozen=True)
class JointAnchor:
    event:object
    context:object
    candidates:tuple
    rejected:tuple
    source:dict
    view:object
    last_completed:int
    prefix:bytes
    features:JointFeatures
    state:str
    @property
    def identity(self):
        return digest(dict(event=self.event.identity,context=self.context.identity,
            candidates=[action_identity(c) for c in self.candidates],rejected=self.rejected,
            source=self.source,view=self.view.identity,last_completed=self.last_completed,
            prefix=self.prefix.hex(),features=self.features.identity,state=self.state))


class JointStreamDriver(StreamDriver):
    def __init__(self,*args,joint_selector_identity,endpoint=None,**kwargs):
        from .joint_protocol import JointEndpoint
        self.endpoint=JointEndpoint() if endpoint is None else endpoint
        if type(self.endpoint) is not JointEndpoint:raise ValueError('explicit joint endpoint required')
        if type(joint_selector_identity) is not str or len(joint_selector_identity)!=64:
            raise ValueError('exact separate joint-selector identity required')
        int(joint_selector_identity,16)
        self.joint_selector_identity=joint_selector_identity
        self.arrived=None;self.prepared=None;self.preparation=None;self._arrival_seal=None
        self._joint_selector=None;self._observer=None;self._active_anchor=None;self._branch=False
        super().__init__(*args,**kwargs)
        if (self.endpoint.max_new_tokens>self.problem.model.profile.max_new_tokens
                or max(self.endpoint.eos_ids)>=self.problem.model.mixer.get_input_embeddings().weight.shape[0]):
            raise ValueError('joint endpoint exceeds actual model capacity')
        if self.updater is None or self.boundary.min_tokens!=1 or len(self.boundary.delimiters)!=1:
            raise ValueError('joint mode needs fixed updater and one-command delimiter transport')

    def _configuration(self):
        return digest(dict(base=super()._configuration(),schema='mmla-joint-stream-v1',
            selector=self.joint_selector_identity,features=JOINT_SCHEMA,endpoint=asdict(self.endpoint),
            timing='arrived-early-support-next-source-event',update='LearningResidual.propose-v1'))

    def _arrival_check(self):
        p=self.problem;a=self.arrived
        if (a is None or a.identity!=self._arrival_seal or p.pending is None or p.pending.identity!=a.attempt
                or p.pending.phi!=a.phi or p.pending.memory!=a.memory or p.pending.reader!=a.reader
                or p.pending.world_clock!=a.read_clock
                or tensor_digest({'initial_rng':p.fast.initial_rng})!=a.initial_rng
                or (p.fast.identity,p.memory.identity,p.reader_identity())!=(a.phi,a.memory,a.reader)
                or p.clock!=a.feedback.arrived_at or self._processed_command!=a.completed_command
                or self._clock.position<a.source_position):
            raise ValueError('sealed support predecessor changed before joint decision')
        if (self._joint_selector is None or getattr(self._joint_selector,'identity',None)!=self.joint_selector_identity):
            raise ValueError('declared joint selector changed')

    def _quiet(self):
        if self.arrived is None:return super()._quiet()
        self._check();self._arrival_check()
        if self._processing or self.problem.memory._pending:
            raise RuntimeError('joint event transition is in flight')

    def arrive(self,feedback,*,selector,source_receipt,observe=None):
        """Trusted source adapter releases actual feedback before calling this."""
        self._check();p=self.problem;s=self._clock
        if (self.arrived is not None or self._processing or p.memory._pending or s._pending is not None
                or s._core or s.stop is not None or s.eof or p.pending is None
                or p.pending.world_clock!=p.world_clock or p.world_clock!=0
                or (p.pending.phi,p.pending.memory,p.pending.reader)!=(p.fast.identity,p.memory.identity,p._reader)):
            raise ValueError('one early support on a published unchanged predecessor required')
        if (type(feedback) is not Feedback or feedback.attempt!=p.pending.identity
                or type(feedback.arrived_at) is not int or not p.clock<feedback.arrived_at<2**63-1
                or type(feedback.source) is not str or not 0<len(feedback.source.encode())<=256
                or type(source_receipt) is not str or len(source_receipt)!=64
                or getattr(selector,'identity',None)!=self.joint_selector_identity
                or not callable(selector) or (observe is not None and not callable(observe))):
            raise ValueError('bound actual arrived support and joint selector required')
        n=p.model.mixer.get_input_embeddings().weight.shape[0]
        if (type(feedback.input_ids) is not tuple or type(feedback.labels) is not tuple
                or not 1<len(feedback.input_ids)<=p.model.profile.local_tokens
                or len(feedback.input_ids)!=len(feedback.labels)
                or any(type(x) is not int or not 0<=x<n for x in feedback.input_ids)
                or any(type(x) is not int or not (x==-100 or 0<=x<n) for x in feedback.labels)
                or all(x==-100 for x in feedback.labels[1:])):
            raise ValueError('bounded finite-vocabulary external supervision required')
        self.arrived=ArrivedFeedback(feedback,p.pending.identity,p.fast.identity,p.memory.identity,p._reader,
            p.world_clock,s.position,self._processed_command,source_receipt,tensor_digest({'initial_rng':p.fast.initial_rng}))
        self._arrival_seal=self.arrived.identity
        self._joint_selector,self._observer=selector,observe;p.clock=feedback.arrived_at
        return self.arrived.identity

    def propose(self,*args,**kwargs):
        if self.arrived is not None:raise RuntimeError('joint pending owns the unresolved support attempt')
        return super().propose(*args,**kwargs)
    def feedback(self,*args,**kwargs):
        raise RuntimeError('joint mode requires arrived feedback and a real next source event')
    def finish_source(self,**kwargs):
        if self.arrived is not None:raise RuntimeError('joint feedback has not reached a real next event')
        return super().finish_source(**kwargs)
    def export_collecting(self):
        if self.arrived is not None or self.prepared is not None:
            raise RuntimeError('joint transactions have no intermediate recovery cut')
        return super().export_collecting()

    def _branch_identity(self):
        return digest(dict(base=super()._branch_identity(),arrived=None if self.arrived is None else self.arrived.identity,
            arrival_seal=self._arrival_seal,
            prepared=None if self.prepared is None else (self.prepared.identity,tensor_digest({'initial_rng':self.prepared.initial_rng})),preparation=self.preparation))

    def _prepare(self):
        self._arrival_check();p=self.problem;a=self.arrived
        if self.prepared is not None:raise ValueError('support computation cannot be repeated')
        # Shadow owns all mutable reads, counters, fast factors and memory. The
        # live event's later world_clock cannot alter arrived feedback visibility.
        q=copy.copy(p);q.cost={k:0. if k=='wall_seconds' else 0 for k in p.cost}
        started=time.monotonic();record=dict(status='RUNNING',arrival=a.identity,
            read_clock=a.read_clock,decision_clock=p.world_clock,before=p.fast.identity)
        self.preparation=record
        try:
            record['stage']='FORK_PREDECESSOR'
            q.fast=p.fast.fork();q.memory=p.memory.fork();q.world_clock=a.read_clock
            q.receipts=copy.deepcopy(p.receipts)
            record['stage']='SUPPORT_GRADIENT'
            prefix,keys,values,tokens=q._reads();fb=a.feedback;device=p.model.rho.a.device
            ids=prefix+list(fb.input_ids);labels=[-100]*len(prefix)+list(fb.labels)
            if len(ids)>p.model.profile.local_tokens:raise ValueError('prepared feedback context capacity exceeded')
            ids=torch.tensor([ids],device=device);labels=torch.tensor([labels],device=device)
            q.cost['forwards']+=1;q.cost['feedback_tokens']+=len(fb.input_ids)
            with torch.enable_grad():
                loss=p.model.loss(ids,torch.ones_like(ids),labels,fast=q.fast,keys=keys,values=values,memory_tokens=tokens)
                # Count attempted gradient computation, even if later rejected.
                q.cost['backward_updates']+=1
                successor,stats=p.model.rho.propose(q.fast,loss,max_steps=p.model.profile.max_fast_steps,**self.updater)
            self.prepared=successor
            record.update(status='PREPARED',after=successor.identity,loss=float(loss.detach()),stats=stats,
                updater=copy.deepcopy(self.updater))
        except BaseException as error:
            record.update(status='FAILED',error=type(error).__name__);raise
        finally:
            q.cost['wall_seconds']+=time.monotonic()-started
            record['cost']=copy.deepcopy(q.cost)
            # process() owns wall time, so avoid counting preparation twice.
            for k in p.cost:
                if k!='wall_seconds':p.cost[k]+=q.cost[k]

    def _features(self,context,candidates):
        p=self.problem;base=action_features(p,context,candidates)
        if self.prepared is None or self.preparation['status']!='PREPARED':raise ValueError('one prepared numerical successor')
        delta=[float((getattr(self.prepared,k).detach()-getattr(p.fast,k).detach()).float().square().mean().sqrt().tanh())
            for k in ('a','b','momentum_a','momentum_b')]
        matrix=[];pairs=[]
        for phi in (0,1):
            for row,action in zip(base.matrix,base.actions):
                matrix.append(row+(float(phi),*(delta if phi else (0.,)*4),math.tanh(self.preparation['loss'])))
                pairs.append((phi,action))
        from .joint_protocol import joint_protocol
        binding=dict(base=base.binding,schema=JOINT_SCHEMA,contract=self.contract,protocol=joint_protocol(self),
            arrival=self.arrived.identity,preparation=digest(self.preparation),prepared=self.prepared.identity,
            selector=self.joint_selector_identity)
        return JointFeatures(tuple(matrix),tuple(pairs),binding)

    def _select_commit(self,event,context,candidates,rejected,receipt,selector,*,source,view,last_completed):
        if self.arrived is None:
            return super()._select_commit(event,context,candidates,rejected,receipt,selector,
                source=source,view=view,last_completed=last_completed)
        self._arrival_check();s=self._clock
        if (len(s._events)!=1 or s._cursor!=0 or source['command']!=self.arrived.completed_command+1
                or source['start']!=self.arrived.source_position+1):
            raise ValueError('next original event only; no intervening M event')
        try:self._prepare()
        finally:receipt['joint_preparation']=copy.deepcopy(self.preparation)
        features=self._features(context,candidates)
        anchor=JointAnchor(event,context,tuple(candidates),tuple(rejected),copy.deepcopy(source),view,last_completed,
            wire(receipt),features,self._branch_identity())
        fixed=anchor.identity
        if self._observer is not None:
            self._active_anchor=anchor
            try:self._observer(self,anchor)
            finally:self._active_anchor=None
            self._check()
            if anchor.identity!=fixed or self._branch_identity()!=anchor.state:
                raise ValueError('teacher mutated common predecessor or joint anchor')
        # Deployment receives immutable numeric features and pair IDs only.
        index,selection=self._joint_selector(features)
        self._check()
        if anchor.identity!=fixed or self._branch_identity()!=anchor.state:
            raise ValueError('joint selector mutated common predecessor')
        self._commit_pair(anchor,index,selection,receipt)

    def _commit_pair(self,anchor,index,selection,receipt):
        p=self.problem;a=self.arrived
        if type(index) is not int or not 0<=index<len(anchor.features.pairs):raise ValueError('one issued action pair required')
        phi,action=anchor.features.pairs[index];mi=index%len(anchor.candidates)
        if action!=action_identity(anchor.candidates[mi]):raise ValueError('pair identity drift')
        record=dict(phase='JOINT_FEEDBACK',attempt=a.attempt,source=a.feedback.source,
            arrived_at=a.feedback.arrived_at,world_clock=a.read_clock,decision_clock=p.world_clock,
            source_receipt=a.source_receipt,arrival=a.identity,phi_before=p.fast.identity,
            memory_before=p.memory.identity,phi='KEEP',memory='PENDING',status='PENDING',
            updater=copy.deepcopy(self.updater),preparation=copy.deepcopy(self.preparation),
            anchor=anchor.identity,pair=index,pair_identity=(phi,action),selection=copy.deepcopy(selection))
        try:
            if phi:
                p.fast=self.prepared.fork();record['phi']='COMMITTED'
            # Original independent M commit; failures do not undo the Phi commit.
            FrameDriver._commit_owned(self,anchor.event,anchor.context,anchor.candidates,mi,
                dict(mode='PDSA_J',pair=index,selection=copy.deepcopy(selection),anchor=anchor.identity),anchor.rejected,receipt)
            record['memory']='NULL' if mi==0 else 'WRITE';record['status']='COMPLETED'
            receipt['commands'].append(copy.deepcopy(anchor.source))
        except BaseException as error:
            p.failed=True;record.update(status='FAILED',error=type(error).__name__);raise
        finally:
            record.update(phi_after=p.fast.identity,memory_after=p.memory.identity)
            p.receipts.append(record);p.pending=None
            self.arrived=None;self.prepared=None;self.preparation=None;self._arrival_seal=None
            self._joint_selector=None;self._observer=None

    def fork_pair(self,anchor,index):
        """Teacher-only full-state fork; completes this core, never live state."""
        if (self._branch or self._active_anchor is not anchor or type(anchor) is not JointAnchor
                or self._branch_identity()!=anchor.state):raise ValueError('current owner-issued joint anchor required')
        original=self.problem;started=time.monotonic();branch=copy.copy(self);p=copy.copy(original)
        branch.problem=p;branch.fork_complete=False
        p.memory=None;p.cost=copy.deepcopy(original.cost);p.receipts=copy.deepcopy(original.receipts)
        branch.branch_cost_base=copy.deepcopy(original.cost);branch.branch_stream_cost_base=copy.deepcopy(self.cost)
        branch.cost=copy.deepcopy(self.cost);branch.receipts=copy.deepcopy(self.receipts)
        branch._branch=True;branch._active_anchor=None;branch._observer=None
        receipt=json.loads(anchor.prefix);error=None
        try:
            branch.rollback=copy.deepcopy(self.rollback);branch._observed=dict(self._observed)
            branch.prepared=self.prepared.fork();branch.preparation=copy.deepcopy(self.preparation)
            p.fast=original.fast.fork();p.segment=copy.deepcopy(original.segment);branch._clock=p.segment
            p.memory=original.memory.fork();branch.fork_complete=True
            branch._commit_pair(anchor,index,dict(mode='OFFLINE_OWN_PAIR',predictive=False),receipt)
            branch._finish_cut(receipt,anchor.view,anchor.last_completed)
        except BaseException as exc:
            error=exc;p.failed=True;receipt.update(status='FAILED',error=type(exc).__name__,
                stage='PAIR_COMMIT' if branch.fork_complete else 'FORK_PREDECESSOR',
                post_phi=p.fast.identity,post_memory=None if p.memory is None else p.memory.identity)
        finally:
            p.cost['wall_seconds']+=time.monotonic()-started
            branch.receipts.append(receipt);p.receipts.append(dict(phase='STREAM',**receipt));branch._processing=False
        unchanged=self._branch_identity()==anchor.state
        if not unchanged:original.failed=True
        if error is not None and not isinstance(error,Exception):
            error.branch=branch;error.live_unchanged=unchanged;raise error
        if not unchanged:raise BranchFailure(branch,RuntimeError('branch mutated live state or RNG'),live_unchanged=False)
        if error is not None:raise BranchFailure(branch,error,live_unchanged=True)
        return branch

    def branch_work(self):
        if not self._branch:raise ValueError('branch-only incremental accounting')
        return dict(runtime={k:self.problem.cost[k]-v for k,v in self.branch_cost_base.items()},
                    stream={k:self.cost[k]-v for k,v in self.branch_stream_cost_base.items()})


def arm_early_feedback(task,driver,*,selector,observe=None):
    """Production gate: only original released support can enter joint pending."""
    from .stream_task import EngineeringStreamTask
    from ..episode_adapter import supervised_sequence
    if type(task) is not EngineeringStreamTask or type(driver) is not JointStreamDriver or task.task.timing!='early':
        raise ValueError('registered early source task and joint driver required')
    answer=task.release_feedback(driver);p=driver.problem
    ids,mask,labels=supervised_sequence(task.task.dataset.formatter,task.task.question(task.task.support_query()),
        answer,device=p.model.rho.a.device,max_tokens=p.model.profile.local_tokens)
    feedback=Feedback(p.pending.identity,'registered-source:'+task.identity,p.clock+1,
        tuple(ids[0].tolist()),tuple(labels[0].tolist()))
    return driver.arrive(feedback,selector=selector,source_receipt=task.identity,observe=observe)
