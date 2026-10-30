"""Token-stream CSBC for the existing authenticated control-language source.

The host supplies tokens and completion markers as they actually arrive. The
boundary detector sees only the current token. A complete command is compiled
only after its completion marker is inside a closed core. Its event belongs to
that completion token, including when its earlier text arrived in a prior core.
Original decoder character anchors remain authority-checked inside the command.
No prior event is re-anchored or submitted a second time.
"""
from dataclasses import asdict, dataclass
import copy
import time
import math
import hashlib
import torch
from ..boundary import Event
from ..construction import Envelope, pin_context, predict, assemble_actions
from ..frames import FrameDriver, action_identity
from ..frontend import tokenize_view, PredictedFrameError
from ..profile import digest, wire
from .adaptive_boundary import AdaptiveSegmentClock, BoundaryConfig


@dataclass(frozen=True)
class StreamLimits:
    source_tokens: int = 4096
    source_commands: int = 128
    segments: int = 64
    observation_bytes: int = 2097152
    envelope_bytes: int = 4096

    def __post_init__(self):
        for name,cap in [('source_tokens',65536),('source_commands',4096),('segments',4096),
                         ('observation_bytes',16777216),('envelope_bytes',16384)]:
            if type(getattr(self,name)) is not int or not 1 <= getattr(self,name) <= cap:
                raise ValueError('bounded stream resource contract required')


@dataclass(frozen=True)
class ObservedToken:
    offset: int
    token: int
    command: int
    start: int
    complete: bool
    envelope: Envelope

    def __post_init__(self):
        if (any(type(x) is not int or not 1 <= x < 2**63-1 for x in (self.offset,self.command,self.start))
                or self.start > self.offset or type(self.token) is not int or self.token < 0 or type(self.complete) is not bool
                or type(self.envelope) is not Envelope):
            raise ValueError('one real token with authenticated observed source coordinates')

    @property
    def identity(self):return digest(asdict(self))


def restore_observation(row):
    if set(row) != {'offset','token','command','start','complete','envelope'}:
        raise ValueError('complete observed-token record required')
    e=copy.deepcopy(row['envelope'])
    e['allowed']=tuple(e['allowed'])
    if e['acl'] is not None:e['acl']=tuple(e['acl'])
    return ObservedToken(row['offset'],row['token'],row['command'],row['start'],row['complete'],Envelope(**e))


class StreamDriver:
    def __init__(self, problem, tokenizer, *, tokenizer_identity, source_identity, boundary, limits,
                 selector_identity, updater=None):
        p=problem
        if (type(boundary) is not BoundaryConfig or type(limits) is not StreamLimits
                or p.pending is not None or p.segment.position or p.segment.number or p.memory._pending):
            raise ValueError('fresh problem and explicit bounded stream definition required')
        for value in (tokenizer_identity,source_identity,selector_identity):
            if type(value) is not str or len(value)!=64:raise ValueError('exact stream dependency identities')
            int(value,16)
        if updater is not None and (type(updater) is not dict or set(updater)!={'learning_rate','momentum','clip'}
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in updater.values())
                or not 0<updater['learning_rate']<=1 or not 0<=updater['momentum']<1 or not 0<updater['clip']<1e6):
            raise ValueError('fixed bounded stream RTT updater required')
        self.updater=copy.deepcopy(updater)
        self.problem,self.tokenizer=problem,tokenizer
        self.tokenizer_identity,self.source_identity=tokenizer_identity,source_identity
        self.boundary,self.limits,self.selector_identity=boundary,limits,selector_identity
        self._tokenizer=tokenizer
        self._embedding_guard=tuple((id(x),x._version) for x in p.model.mixer.get_input_embeddings().parameters())
        self._template=digest(getattr(tokenizer,'chat_template',None))
        p.segment=AdaptiveSegmentClock(p.model.profile,p.problem_id,config=boundary,source_identity=source_identity)
        self._clock=p.segment
        self.contract=self._configuration()
        self._observed={}
        self._resident_wire_bytes=2  # Exact compact-JSON empty array: [].
        self._last=None
        self._unit_start=1
        self._processed_command=0
        self.rollback={}
        self.receipts=[]
        self._processing=False
        self._last_cut=None
        self.cost=dict(observed_tokens=0,frontend_forwards=0,constructor_forwards=0,
                       segments=0,retained_observation_peak_bytes=0)

    def _configuration(self):
        return digest(dict(schema='mmla-observed-command-stream-v1',tokenizer=self.tokenizer_identity,
            source=self.source_identity,problem=self.problem.problem_id,tenant=self.problem.tenant,principal=self.problem.principal,
            selector=self.selector_identity,updater=self.updater,boundary=asdict(self.boundary),
            limits=asdict(self.limits),profile=self.problem.model.profile.identity,template=self._template,
            ownership='completed-command-marker-in-current-core',partial='bounded-original-token-carry'))

    def _check(self):
        self.problem._ready()
        if (self._configuration()!=self.contract or self.tokenizer is not self._tokenizer
                or self.problem.segment is not self._clock
                or tuple((id(x),x._version) for x in self.problem.model.mixer.get_input_embeddings().parameters())!=self._embedding_guard
                or digest(getattr(self.tokenizer,'chat_template',None))!=self._template):
            raise ValueError('stream runtime definition changed')
        self._clock._check()

    def _quiet(self):
        self._check()
        if self._processing or self.problem.pending is not None or self.problem.memory._pending:
            raise RuntimeError('stream transition or task feedback is in flight')

    def feed(self, observation):
        self._quiet();s=self._clock;p=self.problem
        if type(observation) is not ObservedToken:
            raise ValueError('trusted observed token required')
        x=observation
        if self._last is not None and x.offset==self._last.offset:
            if x!=self._last:raise ValueError('source retry changed an accepted observation')
            return s.feed(x.token,offset=x.offset)
        if (x.offset!=s.position+1 or x.offset>self.limits.source_tokens
                or x.command>self.limits.source_commands or s.number>=self.limits.segments
                or x.envelope.tenant!=p.tenant or len(wire(asdict(x.envelope)))>self.limits.envelope_bytes):
            raise ValueError('source order, authority or finite stream allowance exceeded')
        if self._last is None:
            if x.command!=1 or x.start!=1:raise ValueError('first source command must start at one')
        else:
            prev=self._last
            if x.envelope.world_clock < prev.envelope.world_clock:
                raise ValueError('source time moved backward')
            if prev.complete:
                if x.command!=prev.command+1 or x.start!=x.offset:raise ValueError('one next observed command')
            elif x.command!=prev.command or x.envelope!=prev.envelope or x.start!=prev.start:
                raise ValueError('unfinished command changed its authenticated authority')
        if x.envelope.world_clock < p.world_clock:
            raise ValueError('source time preceded an already processed event')
        staged=dict(self._observed);staged[x.offset]=x
        # Accepted rows are frozen ObservedToken/Envelope values. Account for
        # one new row and its comma; full evidence is still serialized at cuts.
        size=self._resident_wire_bytes+len(wire(asdict(x)))+int(bool(self._observed))
        if len(staged)>p.model.profile.local_tokens or size>self.limits.observation_bytes:
            raise ValueError('resident source evidence capacity exceeded')
        result=s.feed(x.token,offset=x.offset)
        if self._last is None or self._last.complete:self._unit_start=x.offset
        self._last=x;self._observed=staged;self._resident_wire_bytes=size
        self.cost['observed_tokens']+=1
        self.cost['retained_observation_peak_bytes']=max(self.cost['retained_observation_peak_bytes'],size)
        return result

    def finish_source(self, *, offset):
        self._quiet()
        if self._last is not None and not self._last.complete:
            raise ValueError('source EOF cannot invent a missing command-completion marker')
        return self._clock.finish_source(offset=offset)

    @torch.no_grad()
    def _compile(self, records):
        """Only a complete, already observed command reaches the bidirectional frontend."""
        p=self.problem;m=p.model
        ids=tuple(x.token for x in records)
        text=self.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        view,encoded,pool=tokenize_view(self.tokenizer,text,max_tokens=m.profile.local_tokens)
        if tuple(encoded[0].tolist()) != ids:
            raise ValueError('observed command must roundtrip under the bound tokenizer')
        embedded=m.mixer.get_input_embeddings()(encoded.to(m.rho.a.device)).detach()
        self.cost['frontend_forwards']+=1
        result=m.frontend(embedded,pool.to(embedded.device))
        events=m.frontend.decode(result,view)
        if len(events)!=1:raise PredictedFrameError('completed source command must produce one valid predicted event')
        ir=events[0];env=records[-1].envelope
        if (ir.source_digest!=hashlib.sha256(text.encode()).hexdigest()
                or not 0<=ir.anchor_char<len(text)
                or any(span is not None and not 0<=span[0]<span[1]<=len(text) for span in ir.spans)
                or (ir.kind!='null' and ir.object_id!=env.object_id)):
            raise PredictedFrameError('predicted event escaped its observed source authority')
        feature=result['event_features'][0].detach()
        return ir,feature,dict(command=records[-1].command,source=env.source_receipt,
             start=records[0].offset,end=records[-1].offset,text_sha256=digest(text),
             source_tokens=len(ids),compiler_tokens=encoded.numel(),prediction=digest(asdict(ir)))

    def _observation_rows(self):
        rows=[asdict(self._observed[k]) for k in sorted(self._observed)]
        if len(wire(rows))!=self._resident_wire_bytes:
            raise ValueError('resident source byte accounting changed')
        return rows

    def _branch_identity(self):
        """Readonly selection must not mutate any owned source or task state."""
        from ..module_training import rng_snapshot
        from ..training import optimizer_digest
        from ..neural import tensor_digest
        p=self.problem;s=self._clock
        return digest(dict(contract=self._configuration(),problem=p.problem_id,tenant=p.tenant,principal=p.principal,
            reader=p.reader_identity(),rho=p.model.rho.identity,phi=p.fast.identity,memory=p.memory.identity,
            initial_rng=tensor_digest({'rng':p.fast.initial_rng}),pending=None if p.pending is None else p.pending.identity,
            clock=p.clock,world_clock=p.world_clock,attempts=p.attempts,closed=p.closed,failed=p.failed,
            last_read=p.last_read_identity,cost=p.cost,receipts=p.receipts,
            candidates=sorted((str(k),action_identity(c)) for k,c in p.memory._pending.items()),
            parents={k:[x.receipt for x in v] for k,v in self.rollback.items()},
            stream=dict(position=s.position,number=s.number,core=s._core,overlap=s._overlap,
                pending=None if s._pending is None else asdict(s._pending),
                events=None if s._events is None else [dict(identity=e.identity) for e in s._events],
                cursor=s._cursor,last=s._last_receipt,exposed_root=s.exposed_root,exposed_after=s.exposed_after,
                carries=[asdict(c) for c in s.carries],stop=s.stop,eof=s.eof,last_token=s._last_token,
                last_feed_stop=s._last_feed_stop,carry_decision=None if s._carry_decision is None else [asdict(c) for c in s._carry_decision],
                carry_receipt=s._carry_receipt,episode=s.episode),
            observations=self._observation_rows(),
            last=None if self._last is None else asdict(self._last),unit_start=self._unit_start,
            processed_command=self._processed_command,last_cut=self._last_cut,processing=self._processing,
            stream_cost=self.cost,stream_receipts=self.receipts,global_rng=optimizer_digest(rng_snapshot())))

    def process(self, selector):
        self._quiet();p=self.problem;s=self._clock
        if s.stop is None:raise RuntimeError('detector has not stopped the observed prefix')
        if getattr(selector,'identity',None)!=self.selector_identity:
            raise ValueError('same declared deployment selector required')
        before=time.monotonic();self._processing=True
        receipt=dict(status='PENDING',segment=s.number+1,events=[],commands=[],
                     pre_phi=p.fast.identity,pre_memory=p.memory.identity,contract=self.contract)
        try:
            view=s.close(p.memory.identity)
            receipt.update(view=view.identity,stop=view.stop,core=[view.core.core_start,view.core.core_end])
            coords=view.observed_coordinates
            if any(pos not in self._observed or self._observed[pos].token!=tok for pos,tok in coords):
                raise ValueError('closed view lost its original observed evidence')
            visible=[self._observed[pos] for pos,_ in coords]
            ends=[x for x in visible if x.complete and x.offset>=view.core.core_start
                  and x.command>self._processed_command]
            if len(ends)>p.model.profile.max_events:
                raise ValueError('closed stream event capacity exceeded')
            if [x.command for x in ends]!=list(range(self._processed_command+1,self._processed_command+1+len(ends))):
                raise ValueError('missing source command; no silent discard')
            compiled=[]
            for end in ends:
                records=[x for x in visible if x.command==end.command]
                if (not records or records[0].offset!=end.start or records[-1]!=end or any(x.complete for x in records[:-1])
                        or any(x.envelope!=end.envelope or x.start!=end.start for x in records)
                        or any(b.offset!=a.offset+1 for a,b in zip(records,records[1:]))):
                    raise ValueError('incomplete or mixed source command')
                ir,feature,source=self._compile(records)
                compiled.append((ir,feature,end,source))
            events=tuple(Event(view.core.identity,end.offset,i,ir.kind,wire(asdict(ir)))
                         for i,(ir,_,end,_) in enumerate(compiled))
            s.register_events(events)
            last_completed=self._processed_command+len(ends)
            keep=tuple(c.identity for c in s.carries if self._observed[c.start].command>last_completed)
            release=tuple((c.identity,'resolved') for c in s.carries if c.identity not in keep)
            unfinished=[x for x in visible if x.offset>=view.core.core_start and x.command>last_completed]
            spans=() if not unfinished else ((unfinished[0].offset,unfinished[-1].offset),)
            s.stage_carry(keep=keep,new_spans=spans,release=release)
            for event,(ir,feature,end,source) in zip(events,compiled):
                p.world_clock=end.envelope.world_clock
                context=pin_context(p.model,p.memory,ir,feature,end.envelope)
                self.cost['constructor_forwards']+=1
                with torch.no_grad():result,binding=predict(p.model,p.memory,context)
                candidates,rejected=assemble_actions(p.model,p.memory,context,result,binding,
                    rollback_rows=tuple(self.rollback.get(end.envelope.object_id,())[:1]))
                self._select_commit(event,context,candidates,rejected,receipt,selector,
                                    source=source,view=view,last_completed=last_completed)
            self._finish_cut(receipt,view,last_completed)
            return copy.deepcopy(receipt)
        except BaseException as error:
            p.failed=True
            receipt.update(status='FAILED',error=type(error).__name__,post_phi=p.fast.identity,post_memory=p.memory.identity)
            raise
        finally:
            p.cost['wall_seconds']+=time.monotonic()-before
            self.receipts.append(receipt);p.receipts.append(dict(phase='STREAM',**receipt))
            self._processing=False

    def _select_commit(self,event,context,candidates,rejected,receipt,selector,*,source,view,last_completed):
        p=self.problem;s=self._clock;pinned=self._branch_identity()
        index,selection=selector(p,context,candidates)
        self._check()
        if p.segment is not s or self._clock is not s or pinned!=self._branch_identity():
            raise ValueError('memory selector changed the pinned task state')
        FrameDriver._commit_owned(self,event,context,candidates,index,selection,rejected,receipt)
        receipt['commands'].append(source)

    def _finish_cut(self,receipt,view,last_completed):
        p=self.problem;s=self._clock
        receipt['carry']=s.expose(p.memory.identity)
        self._processed_command=last_completed
        retained=set(range(s.position-len(s._overlap)+1,s.position+1))
        for c in s.carries:retained.update(range(c.start,c.end+1))
        self._observed={k:v for k,v in self._observed.items() if k in retained}
        self._resident_wire_bytes=len(wire([asdict(self._observed[k]) for k in sorted(self._observed)]))
        self.cost['segments']+=1
        receipt.update(status='COMPLETED',post_phi=p.fast.identity,post_memory=p.memory.identity)
        self._last_cut=view.identity

    def propose(self, prompt, **generation):
        self._quiet();s=self._clock
        if s.stop is not None or s._pending is not None or s._core:
            raise RuntimeError('publish the stopped/partial source prefix before task generation')
        return self.problem.propose(prompt,**generation)

    def feedback(self, feedback):
        self._check()
        if self._processing or self._clock._pending is not None or self._clock._core:
            raise RuntimeError('feedback adaptation requires published source state')
        return self.problem.feedback(feedback,update=self.updater)

    def export_collecting(self):
        self._quiet();s=self._clock
        state=dict(schema='mmla-stream-driver-v1',contract=self.contract,clock=s.export_collecting(),
            observations=self._observation_rows(),
            last=None if self._last is None else asdict(self._last),unit_start=self._unit_start,
            processed_command=self._processed_command,last_cut=self._last_cut,
            cost=copy.deepcopy(self.cost),receipts=copy.deepcopy(self.receipts))
        if len(wire(state))>4*self.limits.observation_bytes:
            raise ValueError('bounded stream evidence export exceeded')
        return state

    def restore_collecting(self, state):
        """Fresh driver only; complete Phi/M and parent history are owned by session."""
        self._quiet()
        if self._last is not None or self.receipts or self._clock.position:
            raise ValueError('restore requires a fresh stream driver')
        if set(state)!=set(self.export_collecting()) or state['schema']!='mmla-stream-driver-v1' or state['contract']!=self.contract:
            raise ValueError('complete exact stream definition required')
        if len(wire(state))>4*self.limits.observation_bytes:raise ValueError('bounded stream snapshot')
        s=AdaptiveSegmentClock.restore_collecting(self.problem.model.profile,config=self.boundary,
             source_identity=self.source_identity,state=state['clock'])
        if s.episode!=self.problem.problem_id:raise ValueError('saved clock belongs to another problem')
        rows=[restore_observation(r) for r in state['observations']]
        if [r.offset for r in rows]!=sorted(set(r.offset for r in rows)):
            raise ValueError('unique ordered observed coordinates required')
        observed={r.offset:r for r in rows}
        required=set(range(s.position-len(s._core)-len(s._overlap)+1,s.position+1))
        for c in s.carries:required.update(range(c.start,c.end+1))
        if (set(observed)!=required or len(rows)>self.problem.model.profile.local_tokens
                or len(wire(state['observations']))>self.limits.observation_bytes
                or any(r.envelope.tenant!=self.problem.tenant or len(wire(asdict(r.envelope)))>self.limits.envelope_bytes for r in rows)):
            raise ValueError('complete bounded original observations required')
        for pos,tok in enumerate(s._overlap+tuple(s._core),start=s.position-len(s._core)-len(s._overlap)+1):
            if observed[pos].token!=tok:raise ValueError('core/overlap source mismatch')
        for c in s.carries:
            if tuple(observed[i].token for i in range(c.start,c.end+1))!=c.tokens:
                raise ValueError('carry original source mismatch')
        last=None if state['last'] is None else restore_observation(state['last'])
        if (s.position==0)!=(last is None) or (last is not None and (last.offset!=s.position
                or (last.offset in observed and observed[last.offset]!=last))):
            raise ValueError('last observation identity mismatch')
        for name in ('unit_start','processed_command'):
            if type(state[name]) is not int:raise ValueError('integral source cursors required')
        if (not 1<=state['unit_start']<=max(1,s.position)
                or not 0<=state['processed_command']<=(0 if last is None else last.command)
                or s.position>self.limits.source_tokens or s.number>self.limits.segments
                or (last is not None and (last.command>self.limits.source_commands or last.start!=state['unit_start']))):
            raise ValueError('source limits/cursor mismatch')
        receipts=state['receipts']
        if (type(receipts) is not list or len(receipts)!=s.number
                or any(x.get('status')!='COMPLETED' or x.get('segment')!=i+1 or x.get('contract')!=self.contract for i,x in enumerate(receipts))
                or (receipts and receipts[-1].get('view')!=state['last_cut'])
                or (not receipts and state['last_cut'] is not None)):
            raise ValueError('successful stream publication history required')
        event_history=[x for receipt in receipts for x in receipt['events']]
        expected_last=None if not event_history else (event_history[-1]['event'],
            'NULL' if event_history[-1]['action']=='NULL' else 'COMMIT')
        if s._last_receipt!=expected_last:raise ValueError('last event acknowledgement lost its actual owner')
        completed=[x for receipt in receipts for x in receipt['commands']]
        if [x['command'] for x in completed]!=list(range(1,state['processed_command']+1)):
            raise ValueError('complete processed-command ownership history')
        # Complete-command records already processed may survive only as overlap.
        # Every unprocessed command must retain its entire observed prefix.
        remaining=[r for r in rows if r.command>state['processed_command']]
        command_ids=sorted(set(r.command for r in remaining))
        expected_end=state['processed_command'] if last is None else last.command
        if command_ids!=list(range(state['processed_command']+1,expected_end+1)):
            raise ValueError('unprocessed command history is incomplete')
        for command in command_ids:
            block=[r for r in remaining if r.command==command]
            end=block[-1]
            if (block[0].offset!=end.start or any(x.start!=end.start or x.envelope!=end.envelope for x in block)
                    or any(x.complete for x in block[:-1])
                    or any(b.offset!=a.offset+1 for a,b in zip(block,block[1:]))
                    or (command!=command_ids[-1] and not end.complete)):
                raise ValueError('unprocessed source lost continuity or authority')
        for a,b in zip(rows,rows[1:]):
            if a.envelope.world_clock>b.envelope.world_clock:
                raise ValueError('restored observed time moved backward')
            if b.offset==a.offset+1:
                if a.complete:
                    if b.command!=a.command+1 or b.start!=b.offset:
                        raise ValueError('restored command transition differs')
                elif b.command!=a.command or b.start!=a.start or b.envelope!=a.envelope:
                    raise ValueError('restored command authority differs')
        if last is not None and s.eof and not last.complete:
            raise ValueError('EOF cannot finish an incomplete command')
        cost=state['cost']
        if (set(cost)!=set(self.cost) or any(type(v) is not int or v<0 for v in cost.values())
                or cost['observed_tokens']!=s.position or cost['segments']!=s.number
                or cost['frontend_forwards']!=len(completed) or cost['constructor_forwards']!=len(completed)):
            raise ValueError('stream work accounting mismatch')
        self.problem.segment=s;self._clock=s;self._observed=observed;self._last=last
        self._resident_wire_bytes=len(wire([asdict(observed[k]) for k in sorted(observed)]))
        self._unit_start=state['unit_start'];self._processed_command=state['processed_command']
        self.receipts=copy.deepcopy(receipts);self._last_cut=state['last_cut'];self.cost=copy.deepcopy(cost)
