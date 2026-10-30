"""Causal adaptive clock with explicit bounded unresolved token fragments.

This clock consumes already tokenized, monotonically numbered observations.
It is separate from the historical complete-message FrameDriver. The delimiter
detector is deterministic, not a learned boundary model. Carries supply context,
never a second event owner or a previously issued event to submit again.
"""
from dataclasses import asdict, dataclass
import copy
from ..boundary import SegmentClock, CompletedView
from ..profile import digest, wire


@dataclass(frozen=True)
class BoundaryConfig:
    min_tokens: int = 32
    max_tokens: int = 256
    delimiters: tuple = ()
    carry_items: int = 4
    carry_bytes: int = 2048
    carry_age: int = 2
    schema: str = 'causal-delimiter-csbc-v1'

    def __post_init__(self):
        object.__setattr__(self, 'delimiters', tuple(self.delimiters))
        if self.schema != 'causal-delimiter-csbc-v1':
            raise ValueError('registered deterministic causal detector required')
        for name, cap in [('min_tokens', 2048), ('max_tokens', 2048), ('carry_items', 64),
                          ('carry_bytes', 65536), ('carry_age', 16)]:
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= cap:
                raise ValueError('finite boundary/carry capacity required')
        if (self.min_tokens > self.max_tokens or len(self.delimiters) > 64
                or len(set(self.delimiters)) != len(self.delimiters)
                or any(type(x) is not int or not 0 <= x < 2**31 for x in self.delimiters)):
            raise ValueError('finite detector definition')

    @property
    def identity(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class Carry:
    source_view: str
    born_segment: int
    start: int
    tokens: tuple

    @property
    def end(self):
        return self.start+len(self.tokens)-1

    @property
    def identity(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class ClosedStreamView:
    core: CompletedView
    carries: tuple
    stop: str
    detector: str

    @property
    def identity(self):
        return digest(dict(core=self.core.identity, carries=[c.identity for c in self.carries],
                           stop=self.stop, detector=self.detector))

    @property
    def observed_coordinates(self):
        """Sparse chronological context; overlap/carry repetitions share an index."""
        seen = {}
        for c in self.carries:
            for i, token in enumerate(c.tokens):
                pos = c.start+i
                if pos in seen and seen[pos] != token:
                    raise ValueError('conflicting carried observations')
                seen[pos] = token
        for pos, token in enumerate(self.core.overlap+self.core.core,
                                    start=self.core.core_start-len(self.core.overlap)):
            if pos in seen and seen[pos] != token:
                raise ValueError('conflicting overlap observations')
            seen[pos] = token
        return tuple(sorted(seen.items()))


class AdaptiveSegmentClock(SegmentClock):
    def __init__(self, profile, episode, *, config, source_identity):
        super().__init__(profile, episode)
        if (type(config) is not BoundaryConfig or config.max_tokens != profile.segment_tokens
                or any(x >= profile.memory_vocab_size for x in config.delimiters)
                or type(source_identity) is not str or len(source_identity) != 64):
            raise ValueError('exact profile/source and bounded detector required')
        int(source_identity, 16)
        self.config, self.source_identity = config, source_identity
        self._definition = digest(dict(profile=profile.identity, config=asdict(config), source=source_identity))
        self.carries = ()
        self.stop = None
        self._carry_decision = None
        self._carry_receipt = None
        self._last_token = None
        self._last_feed_stop = None
        self.eof = False

    def _check(self):
        if digest(dict(profile=self.profile.identity, config=asdict(self.config), source=self.source_identity)) != self._definition:
            raise ValueError('stream definition changed')

    def feed(self, token, *, offset):
        self._check()
        if type(offset) is not int or offset < 1:
            raise ValueError('one-based immutable source token offset')
        if type(token) is not int or not 0 <= token < self.profile.memory_vocab_size:
            raise ValueError('observed vocabulary token required')
        if self._last_token == (offset, token):
            return self._last_feed_stop
        if offset != self.position+1 or self.stop is not None or self.eof or self.position >= 2**63-2:
            raise ValueError('stale source or closed stream prefix')
        if type(token) is not int or not 0 <= token < self.profile.memory_vocab_size:
            raise ValueError('observed vocabulary token required')
        super().observe(token)
        self._last_token = (offset, token)
        if len(self._core) == self.config.max_tokens:
            self.stop = 'MAX_TOKENS'
        elif len(self._core) >= self.config.min_tokens and token in self.config.delimiters:
            self.stop = 'OBSERVED_DELIMITER'
        self._last_feed_stop = self.stop
        return self.stop

    def observe(self, token):
        raise RuntimeError('adaptive input requires feed with immutable source offset')

    def finish_source(self, *, offset):
        self._check()
        if type(offset) is not int or offset != self.position or self._pending is not None:
            raise ValueError('EOF must bind the observed source cursor at a quiescent cut')
        self.eof = True
        if self._core and self.stop is None:
            self.stop = 'SOURCE_EOF'
        return self.stop

    def close(self, predecessor, *, terminal=False):
        self._check()
        if terminal is not False:
            raise ValueError('use authenticated source EOF; no arbitrary terminal flush')
        if (self._pending is not None or not self._core or self.stop is None
                or type(predecessor) is not str or len(predecessor) != 64):
            raise RuntimeError('one actually stopped observed prefix and memory identity required')
        self._validate_carries(self.carries, next_segment=self.number+1)
        view = CompletedView(self.episode, self.number+1, self.position-len(self._core)+1,
                             self.position, self._overlap, tuple(self._core), predecessor)
        closed = ClosedStreamView(view, self.carries, self.stop, self._definition)
        if len(closed.observed_coordinates) > self.profile.local_tokens:
            raise ValueError('carry/overlap/core context overflow; no silent truncation')
        self._pending = view
        return closed

    def _validate_carries(self, carries, *, next_segment):
        if (type(carries) is not tuple or len(carries) > self.config.carry_items
                or sum(len(wire(asdict(c))) for c in carries) > self.config.carry_bytes):
            raise ValueError('carry item/serialized-byte capacity exceeded')
        covered = set()
        for c in carries:
            if (type(c) is not Carry or type(c.tokens) is not tuple or not c.tokens
                    or type(c.start) is not int or c.start < 1 or c.end > self.position
                    or type(c.born_segment) is not int or not 1 <= c.born_segment < next_segment
                    or next_segment-c.born_segment > self.config.carry_age
                    or type(c.source_view) is not str or len(c.source_view) != 64
                    or any(type(x) is not int or not 0 <= x < self.profile.memory_vocab_size for x in c.tokens)):
                raise ValueError('invalid/over-age observed carry; explicit release required')
            coords = set(range(c.start, c.end+1))
            if coords & covered:
                raise ValueError('duplicate carry ownership')
            covered.update(coords)

    def stage_carry(self, *, keep, new_spans=(), release=()):
        """Account for every prior fragment; only current core can create new ones.

        release is ((carry_id, 'resolved'|'abandoned'|'expired'), ...). A reason
        is recorded; exhaustion is never silently converted into a NULL event.
        """
        self._check()
        v = self._pending
        if v is None or self._carry_decision is not None:
            raise RuntimeError('one carry decision for the closed view')
        if type(keep) is not tuple or type(new_spans) is not tuple or type(release) is not tuple:
            raise ValueError('explicit immutable carry decision')
        old = {c.identity:c for c in self.carries}
        drop = dict(release)
        if (len(set(keep)) != len(keep) or len(drop) != len(release) or set(keep)&set(drop)
                or set(keep)|set(drop) != set(old)
                or any(reason not in ('resolved','abandoned','expired') for reason in drop.values())):
            raise ValueError('each old carry must be retained or explicitly released')
        rows = [old[i] for i in keep]
        for lo, hi in new_spans:
            if type(lo) is not int or type(hi) is not int or not v.core_start <= lo <= hi <= v.core_end:
                raise ValueError('new carry must come from this observed core')
            rows.append(Carry(v.identity, v.number, lo, v.core[lo-v.core_start:hi-v.core_start+1]))
        candidate = tuple(sorted(rows, key=lambda c:(c.start,c.identity)))
        self._validate_carries(candidate, next_segment=v.number+1)
        self._carry_decision = candidate
        self._carry_receipt = dict(view=v.identity, kept=list(keep), release=list(release),
                                  created=[c.identity for c in candidate if c.identity not in old])
        return copy.deepcopy(self._carry_receipt)

    def expose(self, successor):
        self._check()
        if self._carry_decision is None or type(successor) is not str or len(successor) != 64:
            raise RuntimeError('explicit carry decision and successor identity required')
        receipt = copy.deepcopy(self._carry_receipt)
        super().expose(successor)
        self.carries = self._carry_decision
        self._carry_decision = self._carry_receipt = None
        self.stop = None
        return receipt

    def root_for(self, next_position):
        self._check()
        if self.stop is not None or self.eof:
            raise ValueError('stopped/finished prefix cannot issue a next-token read')
        return super().root_for(next_position)

    def export_collecting(self):
        """Quiet partial prefix or stopped prefix before any event processing."""
        self._check()
        if self._pending is not None or self._events is not None or self._carry_decision is not None:
            raise RuntimeError('in-flight closed processing is not an automatic resume cut')
        self._validate_carries(self.carries, next_segment=self.number+1)
        return dict(schema='mmla-csbc-collecting-v1', definition=self._definition, episode=self.episode,
            position=self.position, number=self.number, core=list(self._core), overlap=list(self._overlap),
            carries=[asdict(c) for c in self.carries], stop=self.stop, eof=self.eof,
            last_token=self._last_token, last_feed_stop=self._last_feed_stop, exposed_root=self.exposed_root, exposed_after=self.exposed_after,
            last_event_receipt=self._last_receipt)

    @classmethod
    def restore_collecting(cls, profile, *, config, source_identity, state):
        s = copy.deepcopy(state)
        out = cls(profile, s['episode'], config=config, source_identity=source_identity)
        expected = set(out.export_collecting())
        if set(s) != expected or s['schema'] != 'mmla-csbc-collecting-v1' or s['definition'] != out._definition:
            raise ValueError('exact quiet stream schema and definition required')
        for name in ('position','number','exposed_after'):
            if type(s[name]) is not int or not 0 <= s[name] < 2**63-1:
                raise ValueError('bounded stream cursor required')
        core, overlap = tuple(s['core']), tuple(s['overlap'])
        if (len(core) > config.max_tokens or len(overlap) > profile.overlap_tokens
                or any(type(x) is not int or not 0 <= x < profile.memory_vocab_size for x in core+overlap)
                or s['position'] != s['exposed_after']+len(core)
                or type(s['eof']) is not bool or s['number'] > s['exposed_after']
                or s['exposed_after'] > s['number']*config.max_tokens
                or len(overlap) != min(profile.overlap_tokens,s['exposed_after'])
                or (s['number'] == 0 and (overlap or s['exposed_after'] or s['exposed_root'] is not None))
                or (s['number'] > 0 and (not s['exposed_after'] or type(s['exposed_root']) is not str or len(s['exposed_root']) != 64))):
            raise ValueError('inconsistent observed prefix/exposure')
        reason = None
        if len(core) == config.max_tokens:reason = 'MAX_TOKENS'
        elif len(core) >= config.min_tokens and core[-1] in config.delimiters:reason = 'OBSERVED_DELIMITER'
        elif core and s['eof']:reason = 'SOURCE_EOF'
        if s['stop'] != reason or any(core[i] in config.delimiters for i in range(config.min_tokens-1,len(core)-1)):
            raise ValueError('saved detector decision or earliest stopping prefix differs')
        last = None if s['last_token'] is None else tuple(s['last_token'])
        if s['position'] == 0:
            if last is not None:raise ValueError('unexpected observation receipt')
        elif (last is None or len(last) != 2 or type(last[0]) is not int or last[0] != s['position']
              or type(last[1]) is not int or not 0 <= last[1] < profile.memory_vocab_size
              or (core and last[1] != core[-1]) or (not core and overlap and last[1] != overlap[-1])):
            raise ValueError('source retry cursor differs')
        if s['last_feed_stop'] not in (None,'MAX_TOKENS','OBSERVED_DELIMITER'):
            raise ValueError('invalid idempotent feed receipt')
        if core and s['last_feed_stop'] != (None if reason=='SOURCE_EOF' else reason):
            raise ValueError('feed receipt disagrees with current stop')
        out._last_feed_stop=s['last_feed_stop']
        out.position,out.number,out.exposed_after = s['position'],s['number'],s['exposed_after']
        out.exposed_root=s['exposed_root'];out._core=list(core);out._overlap=overlap
        out.stop,out.eof,out._last_token = reason,s['eof'],last
        out.carries=tuple(Carry(c['source_view'],c['born_segment'],c['start'],tuple(c['tokens'])) for c in s['carries'])
        out._validate_carries(out.carries,next_segment=out.number+1)
        if any(c.end > out.exposed_after for c in out.carries):
            raise ValueError('carry source is not published history')
        receipt=s['last_event_receipt']
        if receipt is not None and (type(receipt) not in (list,tuple) or len(receipt)!=2
                or type(receipt[0]) is not str or len(receipt[0])!=64
                or receipt[1] not in ('COMMIT','NULL','FAILED_TERMINAL')):
            raise ValueError('invalid bounded event receipt')
        out._last_receipt=None if receipt is None else tuple(receipt)
        return out
