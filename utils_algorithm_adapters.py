"""Compatibility bridge for legacy imports of algorithm adapter utilities.

The modular test suites expect ``utils_algorithm_adapters`` at repository root.
This shim re-exports the canonical implementations located under ``utils`` so
that both legacy and modular code paths share a single source of truth while
maintaining clean import semantics.
"""
from __future__ import annotations

from utils.utils_algorithm_adapters import (
    register_restored_algorithms,
    RestoredAutoCorrelationConfig,
    RestoredFourierConfig,
    RestoredMetaLearningConfig,
)

__all__ = [
    "register_restored_algorithms",
    "RestoredFourierConfig",
    "RestoredAutoCorrelationConfig",
    "RestoredMetaLearningConfig",
]
