"""Selected synchronous, closed-command-frame production profile.

Every host frame is a completed message of <= segment_tokens, not an unfinished
text prefix. Short messages use the explicit terminal-message flush. This profile
uses zero overlap; it does not claim adaptive CSBC or arbitrary stream extraction.
"""
from dataclasses import dataclass,asdict
import copy,hashlib,json,time
import torch
from .profile import digest,wire
from .frontend import tokenize_view,TypedIR,PredictedFrameError
from .boundary import Event,CompletedView
from .construction import Envelope,pin_context,predict,assemble_actions,commit_decision
from .neural import tensor_digest


@dataclass(frozen=True)
class ObservedCommand:
    text:str
    envelope:Envelope

    def __post_init__(self):
        if type(self.envelope) is not Envelope or type(self.text) is not str or not self.text or '\n' in self.text:
            raise ValueError('one complete command and authenticated source envelope')


@dataclass(frozen=True)
class CompiledFrame:
    problem:str
    predecessor:str
    text:str
    input_ids:tuple[int,...]
    source:tuple[ObservedCommand,...]
    events:tuple[TypedIR,...]
    features:tuple[tuple[float,...],...]
    event_sources:tuple[int,...]
    token_anchors:tuple[int,...]
    frontend:str
    tokenizer:str
    completed_view:str
    prediction:str

    @property
    def identity(self):return digest(asdict(self))


@dataclass(frozen=True)
class DecisionPoint:
    """Ephemeral actual-state teacher anchor; never a deployment feature."""
    frame:CompiledFrame
    events:tuple[Event,...]
    ordinal:int
    context:object
    candidates:tuple
    rejected:tuple
    prefix_receipt:bytes
    state:str

    @property
    def identity(self):
        return digest(dict(frame=self.frame.identity,events=[e.identity for e in self.events],
            ordinal=self.ordinal,context=self.context.identity,
            candidates=[action_identity(c) for c in self.candidates],rejected=self.rejected,
            prefix_receipt=self.prefix_receipt.hex(),state=self.state))


def action_identity(candidate):
    return 'NULL' if candidate is None else digest(dict(predecessor=candidate.predecessor,
        epoch=candidate.epoch,slot=candidate.slot,row=candidate.row.receipt))


class BranchFailure(RuntimeError):
    """Failed replay remains inspectable, including partial commits and costs."""
    def __init__(self,branch,cause,*,live_unchanged):
        super().__init__('offline branch failed: '+type(cause).__name__)
        self.branch=branch;self.cause=cause;self.live_unchanged=live_unchanged


class FrameDriver:
    """One owner-issued compiler result per closed frame; no oracle fallback."""
    def __init__(self,problem,tokenizer,*,tokenizer_identity):
        if problem.model.profile.overlap_tokens!=0:raise ValueError('this explicit message profile uses zero overlap')
        if type(tokenizer_identity) is not str or len(tokenizer_identity)!=64:raise ValueError('verified tokenizer asset identity required')
        self.tokenizer_identity=tokenizer_identity
        self.problem,self.tokenizer=problem,tokenizer;self._issued=None
        self._embedding_versions=tuple((id(x),x._version) for x in problem.model.mixer.get_input_embeddings().parameters())
        self.frame_receipts=[];self.rollback={};self._active_point=None;self._branch=False

    def _quiescent(self):
        p=self.problem;p._ready()
        if self._embedding_versions!=tuple((id(x),x._version) for x in p.model.mixer.get_input_embeddings().parameters()):raise ValueError('frozen frontend embeddings changed')
        if (p.pending is not None or p.segment._pending is not None or p.segment._core
                or p.memory._pending):raise RuntimeError('frame requires a quiescent published predecessor')

    @torch.no_grad()
    def compile(self,commands):
        self._quiescent();p=self.problem;m=p.model
        if self._issued is not None:raise RuntimeError('previous issued frame has not been consumed')
        if (type(commands) is not tuple or not 1<=len(commands)<=m.profile.max_events
                or any(type(c) is not ObservedCommand for c in commands)):
            raise ValueError('complete bounded command frame')
        clocks=[c.envelope.world_clock for c in commands]
        if clocks!=sorted(clocks) or clocks[0]<p.world_clock:raise ValueError('host source clocks must be monotone')
        if any(c.envelope.tenant!=p.tenant for c in commands):raise PermissionError('foreign frame tenant')
        text='\n'.join(c.text for c in commands)
        view,ids,pool=tokenize_view(self.tokenizer,text,max_tokens=m.profile.segment_tokens)
        raw=self.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        if tuple(raw['input_ids'])!=tuple(ids[0].tolist()):raise ValueError('tokenizer mapping drift')
        embedded=m.mixer.get_input_embeddings()(ids.to(m.rho.a.device)).detach()
        result=m.frontend(embedded,pool.to(embedded.device))
        events=m.frontend.decode(result,view)
        # The registered host protocol declares one command per source envelope.
        # This is an output validity check, not a gold count fed to the network.
        if len(events)!=len(commands):raise PredictedFrameError('compiler did not cover the authenticated command frame')
        ranges=[];cursor=0
        for command in commands:ranges.append((cursor,cursor+len(command.text)));cursor=ranges[-1][1]+1
        sources=[];anchors=[]
        for ir in events:
            matching=[i for i,(lo,hi) in enumerate(ranges) if lo<=ir.anchor_char<hi]
            if len(matching)!=1:raise PredictedFrameError('predicted event has no unique source envelope')
            src=matching[0];lo,hi=ranges[src]
            if any(span is not None and not lo<=span[0]<span[1]<=hi for span in ir.spans):
                raise PredictedFrameError('predicted roles cross an authority boundary')
            env=commands[src].envelope
            if ir.kind!='null' and ir.object_id!=env.object_id:raise PredictedFrameError('predicted object outside its source envelope')
            positions=[i for i,(a,b) in enumerate(raw['offset_mapping']) if a<=ir.anchor_char<b]
            if not positions:raise PredictedFrameError('anchor has no observed token position')
            sources.append(src);anchors.append(positions[0])
        if sources!=list(range(len(commands))):raise PredictedFrameError('source envelopes must be covered once in order')
        features=tuple(tuple(result['event_features'][i].detach().cpu().tolist()) for i in range(len(events)))
        tokens=tuple(ids[0].tolist())
        completed=CompletedView(p.problem_id,p.segment.number+1,p.segment.position+1,
            p.segment.position+len(tokens),(),tokens,p.memory.identity)
        frame=CompiledFrame(p.problem_id,p.memory.identity,text,tokens,commands,events,
            features,tuple(sources),tuple(anchors),tensor_digest(m.frontend.state_dict()),
            self.tokenizer_identity,completed.identity,tensor_digest(result))
        self._issued=frame.identity
        return frame

    def branch_identity(self):
        """All mutable task ownership, including issuance and RNG, at an anchor."""
        from .module_training import rng_snapshot
        from .training import optimizer_digest
        p=self.problem;s=p.segment
        return digest(dict(problem=p.problem_id,tenant=p.tenant,principal=p.principal,
            fixed_reader=p._reader,rho=p.model.rho.identity,memory=p.memory.identity,phi=p.fast.identity,
            phi_initial_rng=tensor_digest({'rng':p.fast.initial_rng}),reader=p.reader_identity(),
            pending=None if p.pending is None else p.pending.identity,clock=p.clock,world_clock=p.world_clock,
            attempts=p.attempts,closed=p.closed,failed=p.failed,last_read=p.last_read_identity,
            candidate_pending=sorted((str(k),action_identity(c)) for k,c in p.memory._pending.items()),
            segment=dict(position=s.position,number=s.number,core=s._core,overlap=s._overlap,
                pending=None if s._pending is None else s._pending.identity,
                events=None if s._events is None else [e.identity for e in s._events],cursor=s._cursor,
                last=s._last_receipt,exposed_root=s.exposed_root,exposed_after=s.exposed_after),
            parents={k:[x.receipt for x in v] for k,v in self.rollback.items()},
            receipts=p.receipts,frames=self.frame_receipts,cost=p.cost,issued=self._issued,
            global_rng=optimizer_digest(rng_snapshot())))

    def _record_frame(self,frame,receipt,started,*,observer_seconds=0.):
        p=self.problem
        receipt['successor']=p.memory.identity;receipt['phi']=p.fast.identity
        receipt['offline_observer_seconds']=observer_seconds
        self.frame_receipts.append(receipt);p.receipts.append(dict(phase='FRAME',**receipt))
        self._issued=None;self._active_point=None
        p.cost['wall_seconds']+=max(0.,time.monotonic()-started-observer_seconds)

    def _commit_owned(self,event,context,candidates,index,selection,rejected,receipt):
        """The one production/teacher implementation of every commit side effect."""
        p=self.problem;m=p.model;env=context.envelope
        if type(index) is not int or not 0<=index<len(candidates):raise ValueError('selector returned an unissued action')
        selected=candidates[index];old=None if selected is None else p.memory._rows[selected.slot]
        next_history=dict(self.rollback)
        if selected is not None:
            if old is not None:next_history.pop(old.fields['object_id'],None)
            next_history.pop(env.object_id,None)
            if (old is not None and old.fields['object_id']==env.object_id
                    and old.fields['incarnation']==selected.row.fields['incarnation']
                    and selected.row.status!='purged'):
                next_history[env.object_id]=[old]
        if len(next_history)>m.profile.memory_slots or any(len(rows)!=1 for rows in next_history.values()):
            raise ValueError('bounded immediate-parent store exceeded')
        commit=commit_decision(p.memory,candidates,selected)
        self.rollback=next_history;p.segment.acknowledge(event.identity,'NULL' if selected is None else 'COMMIT')
        p.cost['memory_commits']+=int(selected is not None)
        receipt['events'].append(dict(event=event.identity,context=context.identity,selection=selection,
            rejected=copy.deepcopy(rejected),**commit))

    def _finish(self,frame,events,receipt,selector,*,started,bootstrap=False,observe_context=None,
                observe_decision=None,forced=None):
        p=self.problem;m=p.model;observer_seconds=0.
        start=0 if forced is None else forced[0].ordinal
        try:
            for i in range(start,len(events)):
                event=events[i];env=frame.source[frame.event_sources[i]].envelope;p.world_clock=env.world_clock
                if forced is not None and i==start:
                    point,index=forced;context,candidates,rejected=point.context,point.candidates,point.rejected
                    selection=dict(mode='OFFLINE_OWN_ISSUED_ACTION',anchor=point.identity,predictive=False)
                else:
                    feature=torch.tensor(frame.features[i],device=m.rho.a.device,dtype=torch.float32)
                    context=pin_context(m,p.memory,frame.events[i],feature,env)
                    prior=tuple(self.rollback.get(env.object_id,())[:1])
                    if observe_context is not None:
                        observe_context(context,p.memory.fork(),prior,i)
                        if p.memory.identity!=context.root:raise RuntimeError('observer changed live predecessor')
                    with torch.no_grad():result,binding=predict(m,p.memory,context)
                    candidates,rejected=assemble_actions(m,p.memory,context,result,binding,rollback_rows=prior)
                    if observe_decision is not None and not bootstrap:
                        point=DecisionPoint(frame,events,i,context,candidates,rejected,wire(receipt),self.branch_identity())
                        self._active_point=point;point_identity=point.identity;clock=time.monotonic()
                        try:observe_decision(self,point)
                        finally:observer_seconds+=time.monotonic()-clock;self._active_point=None
                        p._ready()
                        if self.branch_identity()!=point.state or point.identity!=point_identity:
                            raise RuntimeError('offline observer changed live state/anchor')
                    if bootstrap:
                        if len(candidates)<2:
                            receipt['failed_event']=dict(event=event.identity,context=context.identity,
                                rejected=copy.deepcopy(rejected),candidates=len(candidates))
                            raise ValueError('initial construction/qualification failed')
                        index,selection=1,dict(mode='HOST_AUTHORIZED_BOOTSTRAP',predictive=False)
                    else:index,selection=selector(p,context,candidates)
                self._commit_owned(event,context,candidates,index,selection,rejected,receipt)
            p.segment.expose(p.memory.identity);receipt['status']='COMPLETED'
            return receipt
        except Exception as error:
            p.failed=True;receipt.update(status='FAILED',error=type(error).__name__);raise
        finally:self._record_frame(frame,receipt,started,observer_seconds=observer_seconds)

    def fork_action(self,point,index,continuation):
        """Consume a current issued action in an isolated complete-state branch.

        Callable only inside the teacher observer. Branches never inherit that
        observer, and global RNG must stay unchanged (the selected replay is
        deterministic greedy). Remaining future frames/feedback are the trusted
        task adapter's responsibility, not supplied to the continuation selector.
        """
        if (self._branch or self._active_point is not point or type(point) is not DecisionPoint
                or type(index) is not int or not 0<=index<len(point.candidates)
                or self.branch_identity()!=point.state):raise ValueError('one current actual decision point required')
        original=self.problem;original._ready()
        if original.pending is not None or original.segment._cursor!=point.ordinal:
            raise ValueError('teacher anchor is not the current pending event')
        branch=copy.copy(self);branch._branch=True;branch._active_point=None
        branch.rollback=copy.deepcopy(self.rollback);branch.frame_receipts=copy.deepcopy(self.frame_receipts)
        p=copy.copy(original);p.fast=original.fast.fork();p.memory=original.memory.fork()
        p.segment=copy.deepcopy(original.segment);p.receipts=copy.deepcopy(original.receipts)
        p.cost={k:0. if k=='wall_seconds' else 0 for k in original.cost}
        branch.problem=p
        receipt=json.loads(point.prefix_receipt)
        failure=None
        try:
            branch._finish(point.frame,point.events,receipt,continuation,started=time.monotonic(),forced=(point,index))
        except Exception as error:failure=error
        # Both successful and failed replays retain an inspectable owned branch.
        # Mutation is fatal even when the replay itself also failed.
        unchanged=self.branch_identity()==point.state
        if not unchanged:
            raise BranchFailure(branch,RuntimeError('teacher branch changed live task or global RNG'),
                                live_unchanged=False) from failure
        if failure is not None:raise BranchFailure(branch,failure,live_unchanged=True) from failure
        return branch

    def consume(self,frame,selector,*,bootstrap=False,observe_context=None,observe_decision=None):
        """Close/own the frame once; all actions use the same publication path."""
        self._quiescent();p=self.problem;m=p.model;before=time.monotonic()
        if self._branch and observe_decision is not None:raise ValueError('offline branches cannot recursively create teachers')
        if (type(frame) is not CompiledFrame or frame.identity!=self._issued
                or frame.problem!=p.problem_id or frame.predecessor!=p.memory.identity
                or frame.frontend!=tensor_digest(m.frontend.state_dict()) or frame.tokenizer!=self.tokenizer_identity):
            raise ValueError('frame was not issued for this exact current predecessor')
        clocks=[x.envelope.world_clock for x in frame.source]
        if clocks!=sorted(clocks) or not clocks or clocks[0]<p.world_clock:raise ValueError('frame source time is stale')
        if bootstrap and (p.segment.number!=0 or p.memory.version!=0 or any(e.kind!='create' for e in frame.events)):
            raise ValueError('bootstrap is only the first empty-store create frame')
        receipt=dict(frame=frame.identity,predecessor=frame.predecessor,bootstrap=bootstrap,
            source_receipts=[x.envelope.source_receipt for x in frame.source],
            source_frame_number=p.segment.number+1,events=[],status='PENDING')
        try:
            for token in frame.input_ids:p.segment.observe(token)
            view=p.segment.close(p.memory.identity,terminal=True)
            if view.identity!=frame.completed_view:raise ValueError('closed frame position/root changed')
            receipt['completed_view']=view.identity
            events=tuple(Event(view.identity,view.core_start+a,i,ir.kind,wire(asdict(ir)))
                for i,(a,ir) in enumerate(zip(frame.token_anchors,frame.events)))
            p.segment.register_events(events)
        except Exception as error:
            p.failed=True;receipt.update(status='FAILED',error=type(error).__name__)
            self._record_frame(frame,receipt,before);raise
        return self._finish(frame,events,receipt,selector,started=before,bootstrap=bootstrap,
            observe_context=observe_context,observe_decision=observe_decision)
