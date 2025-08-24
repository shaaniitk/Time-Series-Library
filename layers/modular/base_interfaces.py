"""
Lightweight base interface stubs for the modular component architecture.

These abstract base classes provide a minimal, stable surface so that
modules across the codebase can import and subclass them without pulling in
legacy dependencies. They intentionally keep requirements small while
preserving the expected constructor shape and common helper hooks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


class BaseComponent(nn.Module, ABC):
    """Common minimal base class for all modular components.

    Parameters
    ----------
    config: Any
        A configuration object or namespace; stored on ``self.config`` for
        downstream access. Concrete subclasses may expect specific fields.
    """

    def __init__(self, config: Any | None = None) -> None:
        super().__init__()
        self.config = config


class BaseBackbone(BaseComponent, ABC):
    """Abstract backbone contract used by adapters and the registry."""

    @abstractmethod
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover - interface
        """Encode input and return hidden states [B, T, D]."""
        raise NotImplementedError

    @abstractmethod
    def get_d_model(self) -> int:
        """Return the internal model dimension (d_model)."""
        raise NotImplementedError

    def supports_seq2seq(self) -> bool:  # default capability
        return False

    def get_backbone_type(self) -> str:  # default identifier
        return self.__class__.__name__.lower()


class BaseProcessor(BaseComponent, ABC):
    """Processors transform tensors between stages (e.g., wavelet/normalization)."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


class BaseAttention(BaseComponent, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class BaseEmbedding(BaseComponent, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class BaseLoss(BaseComponent, ABC):
    """Loss base with a default forward delegating to compute_loss.

    Subclasses may implement either `forward` directly or a `compute_loss`
    method with signature `(pred, target, *args, **kwargs)` returning a
    torch.Tensor. If neither is provided, instantiation will raise at call.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # pragma: no cover - thin delegator
        compute = getattr(self, "compute_loss", None)
        if callable(compute):
            return compute(pred, target, *args, **kwargs)  # type: ignore[misc]
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward() or compute_loss()"
        )


class BaseOutput(BaseComponent, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class BaseFeedForward(BaseComponent, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


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
