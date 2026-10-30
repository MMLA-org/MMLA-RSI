"""Explicit bounded state and substrate choices for the MiniCPM projection."""
from dataclasses import asdict, dataclass
import hashlib
import json


BASE_REVISION = "a60b37f1fc409c54e1e337b0723aaac6f92dfec0"


def wire(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(wire(value)).hexdigest()


@dataclass(frozen=True)
class ArchitectureProfile:
    """Engineering capacities; no claimed deployment SLA or performance result.

    MiniCPM's 24 original mixer layers stay intact. The paper's example 12-block
    hybrid is a distinct selectable architecture, not an implicit replacement.
    """

    schema: str = "mmla-minicpm-full-v3"
    base_revision: str = BASE_REVISION
    hidden_size: int = 1536
    rank: int = 4
    local_tokens: int = 2048
    segment_tokens: int = 256
    overlap_tokens: int = 32
    max_events: int = 4
    max_candidates: int = 1024
    memory_slots: int = 256
    canonical_bytes: int = 2048
    neural_width: int = 128
    key_width: int = 64
    metadata_bytes: int = 1024
    max_fast_steps: int = 8
    max_new_tokens: int = 256
    max_attempts: int = 8
    top_k: int = 8
    cache_mode: str = "recompute"
    memory_view: str = "neural"
    memory_tokens: int = 0
    memory_vocab_size: int = 130560
    memory_prefix_start: tuple = ()
    memory_prefix_end: tuple = ()
    memory_prefix_separator: tuple = ()
    memory_forbidden_tokens: tuple = ()

    def __post_init__(self):
        if self.schema not in ("mmla-minicpm-full-v3","mmla-minicpm-full-v4") or self.base_revision != BASE_REVISION:
            raise ValueError("unregistered schema or substrate revision")
        for name in ('memory_prefix_start','memory_prefix_end','memory_prefix_separator','memory_forbidden_tokens'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        if (type(self.memory_tokens) is not int or not 0<=self.memory_tokens<=128
                or (self.schema=='mmla-minicpm-full-v3')!=(self.memory_tokens==0)):
            raise ValueError('explicit legacy or token-preserving schema required')
        if type(self.memory_vocab_size) is not int or self.memory_vocab_size<=0:raise ValueError('vocabulary bound')
        if self.memory_tokens:
            if self.memory_view!='neural' or not self.memory_prefix_start or not self.memory_prefix_end:
                raise ValueError('fixed bounded neural-prefix format required')
            for name in ('memory_prefix_start','memory_prefix_end','memory_prefix_separator','memory_forbidden_tokens'):
                ids=getattr(self,name)
                if len(ids)>64 or any(type(x) is not int or not 0<=x<self.memory_vocab_size for x in ids):
                    raise ValueError('finite registered prefix coordinates')
        elif any((self.memory_prefix_start,self.memory_prefix_end,self.memory_prefix_separator,self.memory_forbidden_tokens)):
            raise ValueError('legacy profile cannot carry token channel')
        positive = ("hidden_size", "rank", "local_tokens", "segment_tokens", "max_events",
                    "memory_slots", "canonical_bytes", "neural_width", "key_width",
                    "metadata_bytes", "max_fast_steps", "max_new_tokens", "max_attempts", "top_k", "max_candidates")
        for name in positive:
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("positive integral capacity required: " + name)
        if self.rank != 4 or self.top_k > self.memory_slots:
            raise ValueError("input LARC4 and bounded memory selection required")
        if type(self.overlap_tokens) is not int or not 0 <= self.overlap_tokens <= self.segment_tokens:
            raise ValueError("invalid overlap capacity")
        if self.segment_tokens + self.overlap_tokens > self.local_tokens:
            raise ValueError("completed view exceeds local capacity")
        if self.cache_mode != "recompute" or self.memory_view not in ("canonical", "neural"):
            raise ValueError("undeclared cache or memory view")

    @property
    def identity(self):
        return digest(asdict(self))

    @property
    def neural_value_bytes(self):
        return 4*self.neural_width+(4+4*self.memory_tokens if self.memory_tokens else 0)

    @property
    def row_bytes(self):
        # Fixed padding, FP32 key/value bytes, status byte, lengths and metadata.
        return 1 + 4 + self.canonical_bytes + 4*self.key_width + self.neural_value_bytes + 4 + self.metadata_bytes

    @property
    def resident_bytes(self):
        return self.memory_slots * self.row_bytes
