"""Role-bound ALFWorld observations and verified action-prefix replay.

No environment or model starts at import. A reviewed backend must implement the
protocol below. Actors receive VisibleFrame only; the runner owns receipts,
labels and hidden-state digests. This is an API boundary, not a Python sandbox.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Protocol


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sha(value):
    if type(value) is not str or len(value) != 64:
        raise ValueError('exact SHA256 identity required')
    try:
        if len(bytes.fromhex(value)) != 32:
            raise ValueError('exact SHA256 identity required')
    except ValueError as exc:
        raise ValueError('exact SHA256 identity required') from exc
    if value.lower() != value:
        raise ValueError('canonical lowercase SHA256 required')
    return value


@dataclass(frozen=True)
class TaskPermit:
    task_id: str
    floor: str
    role: str
    game_path: str
    game_sha256: str
    trajectory_path: str
    trajectory_sha256: str
    partition_sha256: str

    def __post_init__(self):
        if self.role not in {'FIT', 'DEV', 'C_J0', 'C_J1', 'C_J2', 'A_RSI1', 'A_RSI2', 'FINAL_OFFICIAL'}:
            raise ValueError('executable role from fixed partition required')
        if any(type(x) is not str or not x for x in (self.task_id, self.floor)):
            raise ValueError('complete physical task identity required')
        for value in (self.game_sha256, self.trajectory_sha256, self.partition_sha256):
            _sha(value)
        for value in (self.game_path, self.trajectory_path):
            if type(value) is not str or not value:
                raise ValueError('original source path required')
            path = PurePosixPath(value)
            if path.is_absolute() or '..' in path.parts or str(path) != value:
                raise ValueError('canonical relative original source path required')

    @property
    def identity(self):
        return _digest(asdict(self))

    @property
    def teacher_allowed(self):
        return self.role == 'FIT'

    def verify_files(self, root):
        """Runner-only exact original files; never return paths to the actor."""
        root = Path(root).resolve(strict=True)
        result = []
        for relative, expected in ((self.game_path, self.game_sha256),
                                   (self.trajectory_path, self.trajectory_sha256)):
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('source escapes the registered data root')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('original game/trajectory identity mismatch')
            result.append(path)
        return tuple(result)


def load_permits(path, *, expected_sha256):
    """Use the exact whole-floor partition, including explicit quarantine."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != _sha(expected_sha256):
        raise ValueError('partition bytes changed')
    allowed = {'FIT', 'DEV', 'C_J0', 'C_J1', 'C_J2', 'A_RSI1', 'A_RSI2', 'FINAL_OFFICIAL'}
    permits, floors, games = {}, {}, set()
    for line in data.splitlines():
        row = json.loads(line)
        if row['role'] == 'EXCLUDED_UNSUPPORTED':
            if row['eligible'] is not False or not row['exclusions']:
                raise ValueError('unsupported exclusion lacks its original reason')
            continue
        if row['role'] == 'QUARANTINE':
            continue
        role, floor, task = row['role'], row['floor_plan'], row['task_id']
        if (role not in allowed or row['eligible'] is not True or row['exclusions']
                or task in permits or row['game_sha256'] in games
                or (floor in floors and floors[floor] != role)):
            raise ValueError('invalid, duplicate or crossing physical task role')
        for name in ('game_path', 'source_path'):
            p = PurePosixPath(row[name])
            if p.is_absolute() or '..' in p.parts or str(p) != row[name]:
                raise ValueError('canonical relative original source path required')
        permit = TaskPermit(task, floor, role, row['game_path'], _sha(row['game_sha256']),
                            row['source_path'], _sha(row['trajectory_sha256']), expected_sha256)
        permits[task] = permit; floors[floor] = role; games.add(permit.game_sha256)
    if not permits:
        raise ValueError('empty registered task partition')
    return permits


@dataclass(frozen=True)
class EpisodeLimits:
    max_steps: int
    max_observation_bytes: int
    max_goal_bytes: int
    max_action_bytes: int
    max_commands: int
    expose_admissible_commands: bool

    def __post_init__(self):
        for name, cap in (('max_steps', 512), ('max_observation_bytes', 65536),
                          ('max_goal_bytes', 16384), ('max_action_bytes', 2048), ('max_commands', 2048)):
            v = getattr(self, name)
            if type(v) is not int or not 1 <= v <= cap:
                raise ValueError('finite registered episode bound: ' + name)
        if type(self.expose_admissible_commands) is not bool:
            raise TypeError('explicit common action-information policy required')


@dataclass(frozen=True)
class VisibleFrame:
    """The entire fresh environment input available to every online actor."""
    goal: str
    observation: str
    admissible_commands: tuple[str, ...]

    def prompt_payload(self):
        # Deliberately no episode/task/floor/path, teacher, facts, plans or won.
        return asdict(self)


@dataclass(frozen=True)
class BackendFrame:
    goal: str
    observation: str
    admissible_commands: tuple[str, ...]
    done: bool
    won: bool
    state_sha256: str


class EpisodeBackend(Protocol):
    """Private runner backend. Hash ALL execution state relevant to replay.

    identity binds code/runtime/wrappers and the complete hidden-state serializer.
    Teacher must be physically absent outside FIT; filesystem split is irrelevant.
    Model/strong-rule code never receives this object or its state hash.
    """
    identity: str
    has_teacher: bool

    def reset(self) -> BackendFrame: ...
    def step(self, action: str) -> BackendFrame: ...
    # Pure read of a precomputed current-state label; it must not advance RNG,
    # environment or expert state. The backend review must verify this property.
    def teacher_action(self) -> str | None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class StepReceipt:
    index: int
    action: str
    before: str
    after: str


@dataclass(frozen=True)
class ReplayCheckpoint:
    permit: TaskPermit
    seed: int
    limits: EpisodeLimits
    backend_identity: str
    initial: str
    steps: tuple[StepReceipt, ...]


@dataclass(frozen=True)
class FeedbackPair:
    """Real observation target paired with its PRE-arrival public context.

    The controller separately pins pre-arrival Phi and M. New observations must
    not be written to that old M or inserted in ``before`` before computing CE.
    """
    before: VisibleFrame
    action: str
    observation_target: str
    after: VisibleFrame
    receipt: StepReceipt
    terminal: bool


class ReplayEpisode:
    def __init__(self, permit, *, seed, limits, factory, on_work):
        if (type(permit) is not TaskPermit or type(seed) is not int or not 0 <= seed < 2**32
                or type(limits) is not EpisodeLimits or not callable(on_work)):
            raise ValueError('bound original task, seed, limits and replay accounting required')
        if permit.role == 'QUARANTINE':
            raise PermissionError('quarantined task cannot be executed')
        self.permit, self.seed, self.limits = permit, seed, limits
        self._factory, self._on_work = factory, on_work
        self._steps, self._closed, self._failed = [], False, False
        self._backend = factory(permit, seed, limits)
        try:
            _sha(self._backend.identity)
            if type(self._backend.has_teacher) is not bool or self._backend.has_teacher != permit.teacher_allowed:
                raise PermissionError('teacher wrapper must follow partition role, not source directory')
            self._backend_identity = self._backend.identity
            self._on_work('reset', 1)
            self._frame = self._validate(self._backend.reset())
            self._initial = self._frame_id(self._frame)
        except BaseException:
            self._failed = True
            self.close()
            raise

    def _validate(self, frame):
        if type(frame) is not BackendFrame or type(frame.done) is not bool or type(frame.won) is not bool:
            raise ValueError('typed backend result required')
        if frame.won and not frame.done:
            raise ValueError('success must terminate the finite task')
        for text, bound in ((frame.goal, self.limits.max_goal_bytes),
                            (frame.observation, self.limits.max_observation_bytes)):
            if type(text) is not str or not text or len(text.encode()) > bound:
                raise ValueError('complete bounded observation/goal, no silent truncation')
        if type(frame.admissible_commands) is not tuple or len(frame.admissible_commands) > self.limits.max_commands:
            raise ValueError('bounded complete action-information tuple required')
        for command in frame.admissible_commands:
            self._action(command)
        if len(set(frame.admissible_commands)) != len(frame.admissible_commands):
            raise ValueError('duplicate admissible commands')
        _sha(frame.state_sha256)
        return frame

    def _action(self, action):
        if type(action) is not str or len(action.encode()) > self.limits.max_action_bytes:
            raise ValueError('complete finite action required')
        # Invalid-but-well-formed environment commands are still executed and
        # scored. Do not replace them with teacher or admissible alternatives.

    def _check(self):
        if (self._closed or self._failed or self._backend.identity != self._backend_identity
                or self._backend.has_teacher != self.permit.teacher_allowed):
            raise RuntimeError('closed, failed or changed episode backend')

    def _frame_id(self, frame):
        return _digest(dict(permit=self.permit.identity, seed=self.seed,
                            backend=self._backend_identity, limits=asdict(self.limits), frame=asdict(frame)))

    @property
    def visible(self):
        self._check()
        f = self._frame
        commands = f.admissible_commands if self.limits.expose_admissible_commands else ()
        return VisibleFrame(f.goal, f.observation, commands)

    @property
    def terminal(self):
        self._check()
        return self._frame.done or len(self._steps) >= self.limits.max_steps

    @property
    def outcome(self):
        """Runner-only complete-task endpoint; not part of an actor's input."""
        self._check()
        if not self.terminal:
            raise RuntimeError('complete task has not ended')
        return dict(won=self._frame.won, steps=len(self._steps),
                    truncated=not self._frame.done and len(self._steps) == self.limits.max_steps)

    def step(self, action):
        self._check()
        if self.terminal:
            raise RuntimeError('terminal task cannot advance')
        self._action(action)
        before, before_id = self.visible, self._frame_id(self._frame)
        try:
            self._on_work('step', 1)
            frame = self._validate(self._backend.step(action))
            if frame.goal != self._frame.goal:
                raise ValueError('task goal changed during execution')
            receipt = StepReceipt(len(self._steps), action, before_id, self._frame_id(frame))
            self._frame = frame; self._steps.append(receipt)
            return FeedbackPair(before, action, frame.observation, self.visible, receipt, self.terminal)
        except BaseException:
            self._failed = True
            self.close()
            raise

    def training_label(self):
        self._check()
        if not self.permit.teacher_allowed or not self._backend.has_teacher:
            raise PermissionError('teacher labels only for FIT; never DEV/calibration/audit/final')
        if self.terminal:
            return None
        self._on_work('teacher', 1)
        action = self._backend.teacher_action()
        if action is not None:
            self._action(action)
            if action not in self._frame.admissible_commands:
                raise ValueError('teacher label does not belong to the actual current state')
        return action

    def checkpoint(self):
        self._check()
        return ReplayCheckpoint(self.permit, self.seed, self.limits, self._backend_identity,
                                self._initial, tuple(self._steps))

    @classmethod
    def replay(cls, checkpoint, *, factory, on_work):
        """Fresh original environment + same actions; verify EVERY transition.

        All resets/steps are charged. A mismatch is a failed branch, never a new
        task or an invitation to search another seed. Model-state forks alone
        cannot satisfy this operation.
        """
        if type(checkpoint) is not ReplayCheckpoint or len(checkpoint.steps) > checkpoint.limits.max_steps:
            raise ValueError('bounded complete replay checkpoint required')
        new = cls(checkpoint.permit, seed=checkpoint.seed, limits=checkpoint.limits,
                  factory=factory, on_work=on_work)
        try:
            if new._backend_identity != checkpoint.backend_identity or new._initial != checkpoint.initial:
                raise ValueError('initial environment or runtime identity mismatch')
            for expected in checkpoint.steps:
                observed = new.step(expected.action)
                if observed.receipt != expected:
                    raise ValueError('environment replay diverged; branch invalid')
            return new
        except BaseException:
            new.close()
            raise

    def close(self):
        if not self._closed:
            self._closed = True
            self._backend.close()
