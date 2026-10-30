"""Explicit logical-world provenance and replay; never relabel original tasks.

The admitted caller pins a complete READY block and its validation closure.
These records carry identity, not permission to open an audit. A native backend
must verify the derived game/provenance and retain its original runtime guard.
No environment, planner or model starts in this module.
"""
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
import hashlib
import json

from mmla_pdsa.full.alfworld_episode import (
    ReplayEpisode, EpisodeLimits, StepReceipt, _digest, _sha)


@dataclass(frozen=True)
class LogicalWorldPermit:
    task_id: str
    floor: str
    role: str
    game_path: str
    game_sha256: str
    provenance_path: str
    provenance_sha256: str
    source_game_sha256: str
    semantic_sha256: str
    validation_sha256: str
    population_sha256: str
    block_sha256: str

    def __post_init__(self):
        if (self.role not in ('DEV', 'A_RSI1', 'A_RSI2')
                or self.task_id != 'logical-' + self.game_sha256
                or type(self.floor) is not str or not self.floor):
            raise ValueError('explicit derived logical-world role and identity required')
        for name in ('game_sha256', 'provenance_sha256', 'source_game_sha256',
                     'semantic_sha256', 'validation_sha256', 'population_sha256', 'block_sha256'):
            _sha(getattr(self, name))
        for name in ('game_path', 'provenance_path'):
            value = getattr(self, name)
            if (type(value) is not str or not value or PurePosixPath(value).is_absolute()
                    or '..' in PurePosixPath(value).parts or str(PurePosixPath(value)) != value):
                raise ValueError('canonical relative derived artifact path required')
        if self.game_path == self.provenance_path:
            raise ValueError('game and provenance are distinct artifacts')

    @property
    def identity(self):
        return _digest(asdict(self))

    @property
    def teacher_allowed(self):
        return False

    def verify_files(self, root):
        root = Path(root).resolve(strict=True)
        paths = []; payloads = []
        for name, sha, cap in ((self.game_path, self.game_sha256, 4*1024**2),
                               (self.provenance_path, self.provenance_sha256, 64*1024)):
            path = (root/name).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file() or not 0 < path.stat().st_size <= cap:
                raise ValueError('bounded derived artifact inside admitted root required')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError('derived artifact identity changed')
            paths.append(path); payloads.append(raw)
        game = json.loads(payloads[0])
        if set(game) != {'pddl_domain', 'pddl_problem', 'grammar'}:
            raise ValueError('derived game contains no inherited walkthrough or solvable flag')
        provenance = json.loads(payloads[1])
        expected = dict(schema='orbit-logical-world-provenance-v1',
            source_game_sha256=self.source_game_sha256, game_sha256=self.game_sha256,
            semantic_sha256=self.semantic_sha256, validation_sha256=self.validation_sha256,
            population_sha256=self.population_sha256, role=self.role, source_floor=self.floor,
            horizon=64, native_validated=True, plan_replay_won=True)
        if (set(provenance) != set(expected) | {'plan_steps'}
                or any(type(provenance[k]) is not type(v) or provenance[k] != v for k,v in expected.items())
                or type(provenance['plan_steps']) is not int or not 1 <= provenance['plan_steps'] <= 64):
            raise ValueError('complete derived validation provenance required')
        # The private plan itself is never exposed through this interface.
        return tuple(paths)


@dataclass(frozen=True)
class LogicalCheckpoint:
    permit: LogicalWorldPermit
    seed: int
    limits: EpisodeLimits
    backend_identity: str
    initial: str
    steps: tuple[StepReceipt, ...]


class LogicalWorldEpisode(ReplayEpisode):
    """Use frozen action/replay semantics with an explicit new provenance type."""
    def __init__(self, permit, *, seed, limits, factory, on_work):
        if (type(permit) is not LogicalWorldPermit or type(seed) is not int or not 0 <= seed < 2**32
                or type(limits) is not EpisodeLimits or limits.max_steps != 64
                or limits.expose_admissible_commands or not callable(on_work) or not callable(factory)):
            raise ValueError('derived 64-step task and common-information policy required')
        self.permit, self.seed, self.limits = permit, seed, limits
        self._factory, self._on_work = factory, on_work
        self._steps, self._closed, self._failed = [], False, False
        self._backend = factory(permit, seed, limits)
        try:
            _sha(self._backend.identity)
            if self._backend.has_teacher is not False:
                raise PermissionError('derived evaluation backend has no teacher')
            self._backend_identity = self._backend.identity
            on_work('reset', 1)
            self._frame = self._validate(self._backend.reset())
            if self._frame.done:
                raise ValueError('validated derived world must start unresolved')
            self._initial = self._frame_id(self._frame)
        except BaseException:
            self._failed = True
            self.close()
            raise

    def checkpoint(self):
        self._check()
        return LogicalCheckpoint(self.permit, self.seed, self.limits, self._backend_identity,
                                 self._initial, tuple(self._steps))

    @classmethod
    def replay(cls, checkpoint, *, factory, on_work):
        if (type(checkpoint) is not LogicalCheckpoint or type(checkpoint.steps) is not tuple
                or len(checkpoint.steps) > checkpoint.limits.max_steps
                or any(type(s) is not StepReceipt for s in checkpoint.steps)):
            raise ValueError('complete logical-world replay checkpoint required')
        new = cls(checkpoint.permit, seed=checkpoint.seed, limits=checkpoint.limits,
                  factory=factory, on_work=on_work)
        try:
            if new._backend_identity != checkpoint.backend_identity or new._initial != checkpoint.initial:
                raise ValueError('derived initial state or runtime differs')
            for expected in checkpoint.steps:
                if new.step(expected.action).receipt != expected:
                    raise ValueError('derived replay diverged')
            return new
        except BaseException:
            new.close()
            raise
