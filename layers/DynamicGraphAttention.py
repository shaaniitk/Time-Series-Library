"""
Dynamic Graph Attention with rolling correlation-based adjacency matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DynamicGraphConstructor(nn.Module):
    """
    Constructs time-varying adjacency matrices using rolling correlations.
    Uses weighted edges instead of binary thresholding for richer information.
    """
    def __init__(self, window_size: int = 20, threshold: float = 0.3, 
                 use_weighted: bool = True, include_self_loops: bool = True):
        super().__init__()
        self.window_size = window_size
        self.threshold = threshold
        self.use_weighted = use_weighted
        self.include_self_loops = include_self_loops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
        Returns:
            Dynamic adjacency matrices [B, L, D, D]
        """
        B, L, D = x.shape
        
        # Pad the input for the rolling window
        padded_x = F.pad(x.transpose(1, 2), (self.window_size - 1, 0))
        
        # Use unfold to create sliding windows
        windows = padded_x.unfold(2, self.window_size, 1)  # [B, D, L, window_size]
        windows = windows.permute(0, 2, 3, 1)  # [B, L, window_size, D]

        # Vectorized correlation computation
        mean = windows.mean(dim=2, keepdim=True)
        centered = windows - mean
        
        # Covariance
        cov = torch.einsum('blwd,blwe->blde', centered, centered) / (self.window_size - 1)
        
        # Standard deviations
        std_dev = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1)).unsqueeze(-1)  # [B, L, D, 1]
        
        # Correlation
        corr = cov / (std_dev * std_dev.transpose(-2, -1) + 1e-8)
        
        # Create adjacency matrix with weighted edges (preserves correlation strength)
        if self.use_weighted:
            # Use ReLU to keep positive correlations, preserving strength information
            adj_matrix = F.relu(torch.abs(corr) - self.threshold)
        else:
            # Binary thresholding (legacy mode)
            adj_matrix = (torch.abs(corr) > self.threshold).float()
        
        # Handle self-loops based on configuration
        if not self.include_self_loops:
            adj_matrix.diagonal(dim1=-2, dim2=-1).zero_()
        
        return adj_matrix

class DynamicGraphAttention(nn.Module):
    """
    Dynamic Graph Attention layer that uses the dynamically constructed graph.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Input features [B, L, D]
            adj: Dynamic adjacency matrices [B, L, D, D]
        Returns:
            Output features [B, L, D]
        """
        B, L, D = h.shape
        Wh = torch.matmul(h, self.W)

        # Prepare for attention mechanism
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        # Masked attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Apply attention
        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        B, L, D, _ = Wh.shape
        Wh_repeated_in_chunks = Wh.repeat_interleave(D, dim=2)
        Wh_repeated_alternating = Wh.repeat(1, 1, D, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        return all_combinations_matrix.view(B, L, D, D, 2 * self.out_features)