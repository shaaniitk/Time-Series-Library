"""FourierBlock component (split from monolithic module)."""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from ..base import BaseAttention
from utils.logger import logger  # type: ignore


class FourierBlock(BaseAttention):
    """1D Fourier block performing learnable transformations in frequency domain."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_heads: int,
        seq_len: int,
        modes: int = 64,
        mode_select_method: str = "random",
    ) -> None:
        super().__init__()
        logger.info(
            "Initializing FourierBlock: %s -> %s modes=%s method=%s", in_channels, out_channels, modes, mode_select_method
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.index = self._get_frequency_modes(seq_len, modes, mode_select_method)
        logger.debug("Selected frequency modes: %s", self.index)
        self.scale = 1 / (in_channels * out_channels)
        head_in = in_channels // n_heads
        head_out = out_channels // n_heads
        mode_count = len(self.index)
        self.weights_real = nn.Parameter(self.scale * torch.rand(n_heads, head_in, head_out, mode_count))
        self.weights_imag = nn.Parameter(self.scale * torch.rand(n_heads, head_in, head_out, mode_count))
        self.output_dim_multiplier = out_channels / in_channels

    def _get_frequency_modes(self, seq_len: int, modes: int, mode_select_method: str) -> List[int]:
        modes = min(modes, seq_len // 2)
        if mode_select_method == "random":
            index = list(range(0, seq_len // 2))
            np.random.shuffle(index)
            index = index[:modes]
        else:
            index = list(range(0, modes))
        index.sort()
        return index

    def _complex_mul1d(
        self, x: torch.Tensor, weights_real: torch.Tensor, weights_imag: torch.Tensor
    ) -> torch.Tensor:
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        weights = torch.complex(weights_real, weights_imag)
        return torch.einsum("bhio,bhio->bho", x, weights)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, None]:
        B, L, D = queries.shape
        H = self.n_heads
        E = D // H
        x = queries.view(B, L, H, E).permute(0, 2, 3, 1)
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            if i >= x_ft.shape[3] or wi >= len(self.index):
                continue
            out_ft[:, :, :, wi] = self._complex_mul1d(
                x_ft[:, :, :, i : i + 1], self.weights_real[:, :, :, wi : wi + 1], self.weights_imag[:, :, :, wi : wi + 1]
            ).squeeze(-1)
        x_out = torch.fft.irfft(out_ft, n=L, dim=-1)
        output = x_out.permute(0, 3, 1, 2).reshape(B, L, -1)
        return output, None

__all__ = ["FourierBlock"]

# --- Registration ---
from ...core.registry import component_registry, ComponentFamily  # noqa: E402

component_registry.register(
    name="FourierBlock",
    component_class=FourierBlock,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        "in_channels": 32,
        "out_channels": 32,
        "n_heads": 4,
        "seq_len": 32,
        "modes": 8,
        "mode_select_method": "random",
    },
)
