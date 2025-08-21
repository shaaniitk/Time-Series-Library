"""Deprecated aggregate adaptive components module.

Re-exporting split adaptive components. Use attention.adaptive.* instead.
"""
from __future__ import annotations
import warnings
from .adaptive.meta_learning_adapter import MetaLearningAdapter  # type: ignore  # noqa: F401
from .adaptive.adaptive_mixture import AdaptiveMixture  # type: ignore  # noqa: F401

_warned = False


def __getattr__(name):  # pragma: no cover
    global _warned
    if name.startswith('__') and name.endswith('__') and name not in globals():
        raise AttributeError(name)
    if not _warned:
        warnings.warn(
            "Importing from adaptive_components is deprecated – use attention.adaptive.*",
            DeprecationWarning,
            stacklevel=2,
        )
        _warned = True
    return globals()[name]


__all__ = ["MetaLearningAdapter", "AdaptiveMixture"]