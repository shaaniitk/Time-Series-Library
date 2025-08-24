"""Compatibility shim for legacy import path `layers.base_interfaces`.

It forwards to `layers.modular.base_interfaces` which contains the current
minimal abstract classes used across the codebase.
"""
from .modular.base_interfaces import (  # noqa: F401
    BaseComponent,
    BaseBackbone,
    BaseProcessor,
    BaseAttention,
    BaseEmbedding,
    BaseLoss,
    BaseOutput,
    BaseFeedForward,
)

__all__ = [
    "BaseComponent",
    "BaseBackbone",
    "BaseProcessor",
    "BaseAttention",
    "BaseEmbedding",
    "BaseLoss",
    "BaseOutput",
    "BaseFeedForward",
]
