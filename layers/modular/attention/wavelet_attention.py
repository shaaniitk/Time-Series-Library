"""Deprecated aggregate wavelet attention module.

Re-exports split wavelet components. Use attention.wavelet.* modules instead.
"""
from __future__ import annotations
import warnings

from .wavelet.wavelet_attention import WaveletAttention  # type: ignore  # noqa: F401
from .wavelet.wavelet_decomposition import WaveletDecomposition  # type: ignore  # noqa: F401
from .wavelet.adaptive_wavelet_attention import AdaptiveWaveletAttention  # type: ignore  # noqa: F401
from .wavelet.multiscale_wavelet_attention import MultiScaleWaveletAttention  # type: ignore  # noqa: F401

_warned = False


def __getattr__(name):  # pragma: no cover
    global _warned
    if name.startswith('__') and name.endswith('__') and name not in globals():
        raise AttributeError(name)
    if not _warned:
        warnings.warn(
            "Importing from wavelet_attention is deprecated – use attention.wavelet.*",
            DeprecationWarning,
            stacklevel=2,
        )
        _warned = True
    return globals()[name]


__all__ = [
    "WaveletAttention",
    "WaveletDecomposition",
    "AdaptiveWaveletAttention",
    "MultiScaleWaveletAttention",
]
