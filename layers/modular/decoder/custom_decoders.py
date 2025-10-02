from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardDecoder(nn.Module):
    """Project decoder embeddings onto the target feature dimension."""

    def __init__(self, d_model: int, output_dim: int = 1) -> None:
        super().__init__()
        hidden_dim = max(1, d_model // 2)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return predictions shaped [batch, sequence, output_dim]."""
        return self.mlp(inputs)


class ProbabilisticDecoder(nn.Module):
    """Predict a mean and standard deviation per target feature."""

    def __init__(self, d_model: int, output_dim: int = 1) -> None:
        super().__init__()
        hidden_dim = max(1, d_model // 2)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU())
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.std_dev_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean and positive std-dev tensors each shaped [batch, sequence, output_dim]."""
        processed = self.mlp(inputs)
        mean = self.mean_head(processed)
        std_dev = F.softplus(self.std_dev_head(processed)) + 1e-6
        return mean, std_dev