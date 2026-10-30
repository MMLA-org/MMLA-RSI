"""Causal completed-segment boundary, ownership and exposure mechanics.

The synchronous profile stalls token production while ordered events commit.
No suffix, target, wall-clock worker completion or teacher argument is accepted.
"""
from dataclasses import dataclass
from .profile import ArchitectureProfile, digest


@dataclass(frozen=True)
class CompletedView:
    episode: str
    number: int
    core_start: int
    core_end: int
    overlap: tuple[int, ...]
    core: tuple[int, ...]
    predecessor: str

    @property
    def identity(self):
        return digest(self.__dict__)


@dataclass(frozen=True)
class Event:
    view: str
    anchor: int
    ordinal: int
    kind: str
    content: bytes

    @property
    def identity(self):
        return digest(dict(view=self.view, anchor=self.anchor, ordinal=self.ordinal,
                           kind=self.kind, content=self.content.hex()))


class SegmentClock:
    def __init__(self, profile: ArchitectureProfile, episode: str):
        if type(episode) is not str or not episode or len(episode.encode()) > 128:
            raise ValueError("bounded episode identity required")
        self.profile, self.episode = profile, episode
        self.position = self.number = 0
        self._core: list[int] = []
        self._overlap: tuple[int, ...] = ()
        self._pending: CompletedView | None = None
        self._events: tuple[Event, ...] | None = None
        self._cursor = 0
        self._last_receipt: tuple[str, str] | None = None
        self.exposed_root: str | None = None
        self.exposed_after = 0

    def observe(self, token: int):
        if self._pending is not None:
            raise RuntimeError("synchronous event publication must finish before next token")
        if type(token) is not int or token < 0:
            raise ValueError("observed token ID required")
        if len(self._core) >= self.profile.segment_tokens:
            raise RuntimeError("close completed segment at its cap")
        self._core.append(token)
        self.position += 1

    def close(self, predecessor: str, *, terminal=False):
        if self._pending is not None or not self._core:
            raise RuntimeError("no new completed core")
        if len(self._core) < self.profile.segment_tokens and terminal is not True:
            raise RuntimeError("short core requires explicit terminal flush")
        self._pending = CompletedView(self.episode, self.number + 1,
            self.position-len(self._core)+1, self.position, self._overlap,
            tuple(self._core), predecessor)
        return self._pending

    def register_events(self, events: tuple[Event, ...]):
        view = self._pending
        if view is None or self._events is not None or type(events) is not tuple:
            raise RuntimeError("one event list per completed view")
        if len(events) > self.profile.max_events:
            raise ValueError("event overflow, not a NULL decision")
        for ordinal, event in enumerate(events):
            if (event.view != view.identity or event.ordinal != ordinal
                    or not view.core_start <= event.anchor <= view.core_end
                    or type(event.content) is not bytes
                    or len(event.content) > self.profile.canonical_bytes):
                raise ValueError("invalid event ownership/order/payload")
        self._events, self._cursor = events, 0

    def acknowledge(self, event_id: str, outcome: str):
        if outcome not in ("COMMIT", "NULL", "FAILED_TERMINAL"):
            raise ValueError("missing/unresolved events are not NULL")
        if self._last_receipt == (event_id, outcome):
            return False
        if self._events is None or self._cursor >= len(self._events):
            raise RuntimeError("no pending owned event")
        if self._events[self._cursor].identity != event_id:
            raise ValueError("stale/out-of-order event")
        self._cursor += 1
        self._last_receipt = event_id, outcome
        return True

    def expose(self, successor: str):
        if self._pending is None or self._events is None or self._cursor != len(self._events):
            raise RuntimeError("publish only after complete ordered event processing")
        old = self._pending
        self.exposed_root, self.exposed_after = successor, old.core_end
        combined = old.overlap + old.core
        self._overlap = combined[-self.profile.overlap_tokens:] if self.profile.overlap_tokens else ()
        self._core.clear()
        self.number += 1
        self._pending = self._events = None
        self._cursor = 0

    def root_for(self, next_position: int):
        if self._pending is not None or next_position != self.position + 1:
            raise ValueError("one current pinned root for the next causal token")
        if self.exposed_root is not None and next_position <= self.exposed_after:
            raise ValueError("successor cannot affect closed-prefix logits")
        return self.exposed_root
