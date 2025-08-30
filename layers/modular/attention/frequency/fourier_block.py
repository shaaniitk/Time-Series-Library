import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import numpy as np

from ..base import BaseAttention

class FourierBlock(BaseAttention):
    """
    A 1D Fourier block that performs learnable transformations entirely in the
    frequency domain. It selects a subset of frequency modes to operate on.
    """
    def __init__(self, d_model: int, n_heads: int, seq_len: int, modes: int = 16, **kwargs):
        super().__init__(d_model, n_heads)
        
        self.modes = min(modes, seq_len // 2)
        self.scale = 1 / (d_model * d_model)
        
        # Select a random subset of frequency modes to keep
        self.mode_indices = self._get_frequency_modes(seq_len, self.modes)
        
        # Complex-valued weights for the selected modes
        self.weights = nn.Parameter(
            self.scale * torch.rand(n_heads, d_model // n_heads, d_model // n_heads, self.modes, dtype=torch.cfloat)
        )

    def _get_frequency_modes(self, seq_len: int, modes: int) -> List[int]:
        """Selects a random subset of frequency modes."""
        indices = list(range(0, seq_len // 2))
        np.random.shuffle(indices)
        return sorted(indices[:modes])

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B, L, D = queries.shape
        H = self.n_heads
        E = D // H

        # Reshape and transform to frequency domain
        x = queries.view(B, L, H, E).permute(0, 2, 3, 1) # [B, H, E, L]
        x_ft = torch.fft.rfft(x, dim=-1) # [B, H, E, L/2+1]
        
        # Initialize output tensor in frequency domain
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)

        # Apply learnable transformation to selected modes
        # einsum: [B, H, E, modes], [H, E, E, modes] -> [B, H, E, modes]
        selected_modes = torch.einsum("bhem,heem->bhem", x_ft[:, :, :, self.mode_indices], self.weights)
        out_ft[:, :, :, self.mode_indices] = selected_modes

        # Transform back to time domain
        output = torch.fft.irfft(out_ft, n=L, dim=-1)
        output = output.permute(0, 3, 1, 2).reshape(B, L, D)

        return output, None

# --- Registration ---
from ...core.registry import component_registry, ComponentFamily  # noqa: E402

component_registry.register(
    name="FourierBlock",
    component_class=FourierBlock,
    component_type=ComponentFamily.ATTENTION,
    test_config={
        "d_model": 32,
        "n_heads": 4,
        "seq_len": 64,
        "modes": 8,
    },
)