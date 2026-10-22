"""MMLA-RSI: finite RTT / RTMA rule recursion."""
from .rules import Language, Rule, parse_rule
from .neural import LearningResidual, FastState
from .evidence import EvidenceMemory, MemoryBounds, Record
from .updater import UpdateBounds
from .generation import Controller, RunConfig
from .journal import Journal

__version__ = "0.1.0"
