"""Internal Chronos / T5 style backbone implementation.

This local implementation replaces legacy imports from ``utils.modular_components``.
It provides a lightweight transformer encoder backbone with the same external
behaviour expected by the wrapper: a ``forward`` method returning
``[batch, seq_len, d_model]`` hidden states and helper accessors.

The design keeps dependencies minimal (pure PyTorch) while preserving the
attribute names consumed elsewhere (``d_model``, ``get_d_model`` etc.).
If a real Chronos / HF model is desired later, a drop‑in replacement class
can be added without changing the registry layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn


@dataclass
class ChronosBackboneConfig:
    d_model: int = 512
    n_heads: int = 8
    num_layers: int = 3
    d_ff: int = 2048
    dropout: float = 0.1
    seq_len: int = 96
    pretrained: bool = False  # kept for interface compatibility
    model_name: str = "local/chronos-minimal"


class ChronosBackbone(nn.Module):
    """Lightweight internal backbone approximating a Chronos style encoder.

    It intentionally mirrors a subset of the old Chronos backbone surface so the
    existing wrapper + registry remain stable while removing the cross‑package
    dependency on ``utils.modular_components``.
    """

    def __init__(self, config: ChronosBackboneConfig):  # type: ignore[override]
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.layer_norm = nn.LayerNorm(config.d_model)

    # --- API expected by higher layers --- #
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        """Encode a batch.

        Args:
            x: [batch, seq_len, d_model]
            attention_mask: Optional bool mask (True indicates pad) shape [batch, seq_len]
        """
        if attention_mask is not None:
            # nn.Transformer expects src_key_padding_mask with True for pads
            encoded = self.encoder(x, src_key_padding_mask=attention_mask)
        else:
            encoded = self.encoder(x)
        return self.layer_norm(encoded)

    # Compatibility helpers
    def get_d_model(self) -> int:  # noqa: D401
        return self.d_model

    def get_hidden_size(self) -> int:
        return self.d_model

    def supports_seq2seq(self) -> bool:
        return True

    def get_backbone_type(self) -> str:
        return "chronos"

__all__ = ["ChronosBackbone", "ChronosBackboneConfig"]
