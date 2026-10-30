"""Unvalidated derived input, distinct from an execution-ready task permit."""
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
import hashlib
import json

from mmla_pdsa.full.alfworld_episode import TaskPermit, _digest, _sha
from placement_domain import compile_domain, instantiate


@dataclass(frozen=True)
class ValidationCandidate:
    source: TaskPermit
    choices: tuple[int, ...]
    game_path: str
    game_sha256: str

    def __post_init__(self):
        if type(self.source) is not TaskPermit or self.source.role != 'FIT':
            raise ValueError('an unchanged original FIT source is required')
        if (type(self.choices) is not tuple or not self.choices
                or len(self.choices) > 4096
                or any(type(x) is not int or x < 0 for x in self.choices)):
            raise ValueError('finite complete placement choices required')
        p = PurePosixPath(self.game_path)
        if (not self.game_path or p.is_absolute() or '..' in p.parts
                or str(p) != self.game_path):
            raise ValueError('canonical derived game path required')
        _sha(self.game_sha256)

    @property
    def identity(self):
        return _digest(asdict(self))

    @classmethod
    def from_dict(cls, value):
        return cls(TaskPermit(**value['source']), tuple(value['choices']),
                   value['game_path'], value['game_sha256'])

    def verify(self, source_root, derived_root):
        """Check exact source and reproduce the placement transformation."""
        source_game, trajectory = self.source.verify_files(source_root)
        if source_game.stat().st_size > 4*1024**2 or trajectory.stat().st_size > 4*1024**2:
            raise ValueError('bounded original source required')
        root = Path(derived_root).resolve(strict=True)
        path = (root/self.game_path).resolve(strict=True)
        if (not path.is_relative_to(root) or not path.is_file()
                or not 0 < path.stat().st_size <= 4*1024**2):
            raise ValueError('bounded derived game inside the admitted root required')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.game_sha256:
            raise ValueError('unvalidated game identity changed')
        domain = compile_domain(source_game.read_bytes(), expected_sha256=self.source.game_sha256)
        expected, receipt = instantiate(domain, self.choices)
        game = json.loads(raw)
        if game != expected:
            raise ValueError('derived game does not equal the declared source transformation')
        data = json.loads(trajectory.read_bytes())
        params = data['pddl_params']
        if (type(params) is not dict or len(params) > 16
                or any(type(k) is not str or type(v) not in (str, int, bool, type(None))
                       for k, v in params.items())
                or type(data['task_type']) is not str):
            raise ValueError('bounded source goal parameters required')
        return game, {**params, 'task_type': data['task_type']}, receipt
