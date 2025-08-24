"""Base layer interfaces used by tests to validate type hints.

These are lightweight abstract stubs that define the forward signatures
expected by the component tests. Concrete implementations can inherit
from these classes, but tests only import them to verify annotations.
"""
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn


class BaseEncoderLayer(nn.Module):
    """Minimal encoder layer base with typed forward signature."""

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # noqa: D401
        """Encode sequence x with an optional attention mask.

        Returns a tuple of (encoded_x, attention_weights_optional).
        """
        raise NotImplementedError


class BaseDecoderLayer(nn.Module):
    """Minimal decoder layer base with typed forward signature."""

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
        """Decode sequence x attending over cross, with optional masks.

        Returns a tuple of (decoded_x, attention_weights_or_state).
        """
        raise NotImplementedError
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseEncoderLayer(nn.Module, ABC):
    """Minimal abstract base for encoder layers with expected type hints."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return encoded features and optional attention weights."""
        raise NotImplementedError


class BaseDecoderLayer(nn.Module, ABC):
    """Minimal abstract base for decoder layers with expected type hints."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return seasonal and trend (or two-tensor tuple) outputs."""
        raise NotImplementedError
