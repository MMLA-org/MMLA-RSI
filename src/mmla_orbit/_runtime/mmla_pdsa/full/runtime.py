"""One unresolved problem: full-vocabulary attempts, real feedback, two commits.

Feedback is supplied by a trusted task adapter, not inferred from the model's own
confidence. This local controller is serialized. Receipts prove execution order,
not feedback correctness, functional effect or performance advantage.
"""
from dataclasses import asdict, dataclass
import struct
import time
import torch

from .memory import Memory, Proposal, Authority
from .neural import tensor_digest
from .profile import digest
from .boundary import SegmentClock, Event


@dataclass(frozen=True)
class Attempt:
    number: int
    problem: str
    phi: str
    memory: str
    reader: str
    prompt: tuple[int, ...]
    output: tuple[int, ...]
    clock: int
    world_clock: int = 0

    @property
    def identity(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class Feedback:
    attempt: str
    source: str
    arrived_at: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


class Problem:
    def __init__(self, model, memory: Memory, *, problem_id, tenant, principal,
                 seed, encode_canonical=None, read_tokens=None, world_clock=0, view_reader=None):
        if not all(type(x) is str and 0 < len(x.encode()) <= 128 for x in (problem_id, tenant, principal)):
            raise ValueError("bounded problem/tenant/principal required")
        if memory.profile != model.profile:
            raise ValueError("model/memory profile mismatch")
        self.model, self.memory = model, memory
        self.problem_id, self.tenant, self.principal = problem_id, tenant, principal
        self.fast = model.rho.begin(seed)
        self.encode_canonical = encode_canonical
        self.read_tokens = read_tokens
        self.view_reader=view_reader
        self.last_read_identity=None
        if type(world_clock) is not int or not 0 <= world_clock < 2**63-1:
            raise ValueError('bounded trusted read clock required')
        self.world_clock = world_clock
        self.clock = self.attempts = 0
        self.pending: Attempt | None = None
        self.receipts = []
        self.closed = self.failed = False
        self.segment = SegmentClock(model.profile, problem_id)
        self.cost = dict(forwards=0, backward_updates=0, generated_tokens=0, feedback_tokens=0,
                         memory_reads=0, memory_commits=0, wall_seconds=0.0)
        self._reader = self.reader_identity()

    def reader_identity(self):
        # Hash only the reader, once per registered tensor mutation. This cache
        # holds identity bytes, never Phi/M-dependent activations or logits.
        tensors={k:v for k,v in self.model.state_dict(keep_vars=True).items() if not k.startswith(('mixer.','rho.','risk.','joint_risk.'))}
        guard=tuple((k,id(v),v._version,v.device,v.dtype,tuple(v.shape),tuple(v.stride()),v.storage_offset(),v.data_ptr()) for k,v in tensors.items())
        saved=getattr(self,'_reader_digest_cache',None)
        if saved is None or saved[0]!=guard:
            saved=(guard,tensor_digest(tensors));self._reader_digest_cache=saved
        return digest(dict(substrate=self.model.substrate_identity, profile=self.model.profile.identity,
            view_reader=None if self.view_reader is None else self.view_reader.identity,parameters=saved[1]))

    def _ready(self):
        if self.closed or self.failed:
            raise RuntimeError("problem is closed/failed")
        self.model.rho.factors(self.fast)
        if self.reader_identity() != self._reader:
            raise RuntimeError("reader changed during unresolved problem")

    def _reads(self):
        if self.view_reader is not None:
            from .typed_reads import ReadSet
            result=self.view_reader.read(self.memory,tenant=self.tenant,principal=self.principal,
                clock=self.world_clock,view=self.model.profile.memory_view)
            if (type(result) is not ReadSet or (result.predecessor,result.tenant,result.principal,result.world_clock,result.view,result.reader)
                    != (self.memory.identity,self.tenant,self.principal,self.world_clock,self.model.profile.memory_view,self.view_reader.identity)):
                raise ValueError('typed read must bind current root, scope, clock and exact view')
            if len(result.rows)>self.model.profile.memory_slots:raise ValueError('derived memory capacity')
            if result.view=='canonical' and any(r.key is not None or r.value is not None for r in result.rows):raise ValueError('canonical read must not carry neural tensors')
            values=[r.canonical if result.view=='canonical' else (r.key,r.value) for r in result.rows]
            if result.view=='neural' and any(k is None or v is None for k,v in values):raise ValueError('canonical bytes are not a neural read')
            self.last_read_identity=result.identity
        else:
            selectors = self.memory.selectors(tenant=self.tenant, principal=self.principal,
                                              clock=self.world_clock, view=self.model.profile.memory_view)
            values = [self.memory.read(s, principal=self.principal, clock=self.world_clock) for s in selectors]
        if any(v is None for v in values):
            raise RuntimeError("typed read changed while pinning")
        self.cost['memory_reads'] += len(values)
        if self.model.profile.memory_view == 'canonical':
            if values and self.encode_canonical is None:
                raise ValueError("canonical profile requires explicit trusted serialization/tokenization")
            tokens = []
            for v in values:
                tokens.extend(self.encode_canonical(v))
                if len(tokens) > self.model.profile.local_tokens:
                    raise ValueError("canonical memory read exceeds local context")
            return tokens, None, None, None
        if not values:
            return [], None, None, None
        from .token_memory import memory_tensors
        tensors=memory_tensors(self.model.profile,values,device=self.model.rho.a.device)
        return [],tensors[0],tensors[1],None if len(tensors)==2 else tensors[2]

    def advance_world_clock(self, clock):
        """Trusted adapter time; attempts and feedback order never imply TTL time."""
        self._ready()
        if self.pending is not None or self.segment._pending is not None:
            raise RuntimeError('change read clock only between pinned operations')
        if type(clock) is not int or not self.world_clock <= clock < 2**63-1:
            raise ValueError('read clock must be monotone and bounded')
        self.world_clock = clock

    def propose(self, prompt, *, eos_ids, max_new_tokens=None):
        self._ready()
        if self.pending is not None or self.attempts >= self.model.profile.max_attempts:
            raise RuntimeError("feedback required or attempt budget exhausted")
        if not prompt or any(type(v) is not int or v < 0 for v in prompt):
            raise ValueError("observed prompt tokens required")
        before = time.monotonic()
        self.attempts += 1
        self.clock += 1
        # A failed forward still consumes this attempt. Token work is recorded by
        # callbacks inside generate, including work preceding an exception.
        def forward_started(): self.cost['forwards'] += 1
        def token_generated(): self.cost['generated_tokens'] += 1
        try:
            if self.segment._pending is not None:
                raise RuntimeError("unpublished completed segment")
            tokens, keys, values, memory_tokens = self._reads()
            ids = tuple(tokens) + tuple(prompt)
            memory, phi = self.memory.identity, self.fast.identity
            tensor = torch.tensor([ids], device=self.model.rho.a.device)
            out = self.model.generate(tensor, fast=self.fast, keys=keys, values=values, memory_tokens=memory_tokens,
                eos_ids=eos_ids, max_new_tokens=max_new_tokens,
                on_forward=forward_started, on_token=token_generated)[0].tolist()
            if (memory, phi) != (self.memory.identity, self.fast.identity):
                raise RuntimeError("state changed inside a pinned attempt")
            attempt = Attempt(self.attempts, self.problem_id, phi, memory, self._reader,
                              ids, tuple(out), self.clock, self.world_clock)
            self.pending = attempt
            return attempt
        except Exception as error:
            self.failed = True
            self.receipts.append(dict(phase='PROPOSAL', attempt=self.attempts,
                status='FAILED', error=type(error).__name__))
            raise
        finally:
            self.cost['wall_seconds'] += time.monotonic()-before

    def feedback(self, feedback: Feedback, *, update=None, memory_action=None):
        """Commit Phi first, then optionally one M action. No implicit joint rollback.

        memory_action is a trusted manager callback receiving only current memory
        and the arrived feedback. A predictor adapter must perform its own pinned
        CausalDecision/calibration checks. It is not silently called a learned Q.
        """
        self._ready()
        if (self.pending is None or feedback.attempt != self.pending.identity
                or type(feedback.arrived_at) is not int or feedback.arrived_at <= self.pending.clock
                or type(feedback.source) is not str or not 0 < len(feedback.source.encode()) <= 256):
            raise ValueError("feedback must bind one pending attempt and arrive after it")
        if (len(feedback.input_ids) != len(feedback.labels) or not feedback.input_ids
                or len(feedback.input_ids) > self.model.profile.local_tokens):
            raise ValueError("bounded external feedback labels required")
        record = dict(attempt=self.pending.identity, source=feedback.source, world_clock=self.world_clock,
                      phi_before=self.fast.identity, memory_before=self.memory.identity,
                      phi='NO_UPDATE', memory='NULL', status='PENDING')
        before = time.monotonic()
        self.clock = feedback.arrived_at
        try:
            if update is not None:
                # Memory is re-read under the same permissions. For canonical
                # view, caller's labels are extended with unscored memory tokens.
                prefix, keys, values, memory_tokens = self._reads()
                ids = prefix + list(feedback.input_ids)
                labels = [-100]*len(prefix) + list(feedback.labels)
                device = self.model.rho.a.device
                ids = torch.tensor([ids], device=device)
                labels = torch.tensor([labels], device=device)
                self.cost['forwards'] += 1
                self.cost['feedback_tokens'] += len(feedback.input_ids)
                loss = self.model.loss(ids, torch.ones_like(ids), labels, fast=self.fast, keys=keys, values=values, memory_tokens=memory_tokens)
                new, stats = self.model.rho.propose(self.fast, loss,
                    max_steps=self.model.profile.max_fast_steps, **update)
                self.fast = new
                record['phi'], record['update'] = 'COMMITTED', stats
                record['updater'] = dict(update)
                self.cost['backward_updates'] += 1
            if memory_action is not None:
                candidate = memory_action(self.memory, feedback)
                result = self.memory.commit(candidate)
                record['memory'] = result['action']
                self.cost['memory_commits'] += int(candidate is not None)
            record['status'] = 'COMPLETED'
        except Exception as error:
            self.failed = True
            record['status'], record['error'] = 'FAILED', type(error).__name__
            raise
        finally:
            record.update(phi_after=self.fast.identity, memory_after=self.memory.identity)
            self.receipts.append(record)
            self.pending = None
            self.cost['wall_seconds'] += time.monotonic()-before
        return record

    def close_segment(self, observed_tokens, extract, manage, *, terminal=False):
        """Process observed tokens already issued by the task driver.

        extract sees only the closed core plus bounded read-only overlap. Each
        event's manager sees its own predecessor; root exposure waits for all.
        """
        self._ready()
        before = time.monotonic()
        try:
            for token in observed_tokens:
                self.segment.observe(token)
            view = self.segment.close(self.memory.identity, terminal=terminal)
            events = tuple(extract(view))
            self.segment.register_events(events)
            for event in events:
                candidate = manage(self.memory, view, event)
                self.memory.commit(candidate)
                self.segment.acknowledge(event.identity, 'NULL' if candidate is None else 'COMMIT')
                self.cost['memory_commits'] += int(candidate is not None)
            self.segment.expose(self.memory.identity)
            return view
        except Exception as error:
            self.failed = True
            self.receipts.append(dict(phase='SEGMENT',status='FAILED',error=type(error).__name__))
            raise
        finally:
            self.cost['wall_seconds'] += time.monotonic()-before

    def reset_phi(self):
        self._ready()
        if self.pending is not None:
            raise RuntimeError("cannot reset while awaiting feedback")
        self.fast = self.model.rho.reset(self.fast)

    def complete_query(self, attempt_identity):
        """Resolve a terminal generated answer without inventing feedback.

        The task scorer is outside this interface; no query target, label or
        grade enters state. This records the existing output and closes it.
        """
        self._ready()
        if self.pending is None or self.pending.identity!=attempt_identity:
            raise ValueError('exact own pending terminal query required')
        if (self.segment._pending is not None or self.segment._core or self.segment._events is not None
                or self.memory._pending or (self.pending.phi,self.pending.memory,self.pending.world_clock,self.pending.reader)
                != (self.fast.identity,self.memory.identity,self.world_clock,self._reader)):
            raise RuntimeError('terminal answer requires fully published matching state')
        self.receipts.append(dict(phase='TERMINAL_QUERY',attempt=attempt_identity,
            status='COMPLETED_WITHOUT_FEEDBACK',phi=self.fast.identity,memory=self.memory.identity))
        self.pending=None
        return self.finish()

    def finish(self):
        if self.pending is not None:
            raise RuntimeError("pending feedback must be explicitly resolved before close")
        self.closed = True
        return dict(problem=self.problem_id, receipts=list(self.receipts), cost=dict(self.cost),
                    memory=self.memory.identity, phi=self.fast.identity)
