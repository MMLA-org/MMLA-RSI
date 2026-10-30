"""Unified MMLA engineering implementation, separate from historical experiments.

The package name denotes the intended specification. Consult orchestration/mainline/2026-10-15-minicpm-full-architecture/DELIVERY_PLAN.md for
actual implementation/qualification; an import is not a completed model release.
"""

from .profile import ArchitectureProfile

__all__ = ["ArchitectureProfile"]
