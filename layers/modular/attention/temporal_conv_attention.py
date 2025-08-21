"""Deprecated aggregate temporal convolution attention module.

Re-exports split temporal convolution components. Use attention.temporal_conv.* instead.
"""
from __future__ import annotations
import warnings

from .temporal_conv.causal_convolution import CausalConvolution  # type: ignore  # noqa: F401
from .temporal_conv.temporal_conv_net import TemporalConvNet  # type: ignore  # noqa: F401
from .temporal_conv.convolutional_attention import ConvolutionalAttention  # type: ignore  # noqa: F401

_warned = False


def __getattr__(name):  # pragma: no cover
    global _warned
    if name.startswith('__') and name.endswith('__') and name not in globals():
        raise AttributeError(name)
    if not _warned:
        warnings.warn(
            "Importing from temporal_conv_attention is deprecated – use attention.temporal_conv.*",
            DeprecationWarning,
            stacklevel=2,
        )
        _warned = True
    return globals()[name]


__all__ = [
    "CausalConvolution",
    "TemporalConvNet",
    "ConvolutionalAttention",
]
