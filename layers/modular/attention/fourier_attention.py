"""Deprecated aggregate Fourier attention module.

Re-exports split Fourier components. Use attention.fourier.<module> imports.
Will be removed after migration phase.
"""
from __future__ import annotations
import warnings

from .fourier.fourier_attention import FourierAttention  # type: ignore  # noqa: F401
from .fourier.fourier_block import FourierBlock  # type: ignore  # noqa: F401
from .fourier.fourier_cross_attention import FourierCrossAttention  # type: ignore  # noqa: F401

_warned = False

def __getattr__(name):  # pragma: no cover
    global _warned
    # Avoid intercepting import machinery special attributes (e.g. __path__, __spec__)
    # Returning AttributeError lets Python fall back gracefully instead of KeyError.
    if name.startswith('__') and name.endswith('__') and name not in globals():
        raise AttributeError(name)
    if not _warned:
        warnings.warn(
            "Importing from fourier_attention is deprecated – use attention.fourier.*",
            DeprecationWarning,
            stacklevel=2,
        )
        _warned = True
    return globals()[name]

__all__ = ["FourierAttention", "FourierBlock", "FourierCrossAttention"]
