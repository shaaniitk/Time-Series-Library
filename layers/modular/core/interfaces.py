"""Unified protocol interfaces for modular components.

These Protocols define minimal forward signatures used for runtime validation
inside the unified registry/factory. They deliberately avoid heavy dependencies
so that lightweight mock components can satisfy them in tests.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable, Optional, Tuple
import torch

Tensor = torch.Tensor

@runtime_checkable
class AttentionLike(Protocol):
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, attn_mask: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Optional[Tensor]]: ...

@runtime_checkable
class EncoderLike(Protocol):
    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, list]: ...

@runtime_checkable
class DecoderLike(Protocol):
    def forward(self, x: Tensor, cross: Tensor, x_mask: Optional[Tensor] = None, cross_mask: Optional[Tensor] = None, trend: Optional[Tensor] = None): ...  # flexible tuple

@runtime_checkable
class DecompositionLike(Protocol):
    def forward(self, x: Tensor): ...  # returns (seasonal, trend)

@runtime_checkable
class SamplingLike(Protocol):
    def forward(self, forward_fn, *args, **kwargs): ...

@runtime_checkable
class OutputHeadLike(Protocol):
    def forward(self, x: Tensor): ...

@runtime_checkable
class LossLike(Protocol):
    def forward(self, preds: Tensor, targets: Tensor, **kwargs): ...

CAPABILITY_FLAGS = (
    'provides_trend',
    'requires_attention',
    'requires_decomposition',
)
