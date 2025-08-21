"""GraphAttentionLayer split from graph_attention."""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer supporting batched 3D/4D inputs.

    If input is 4D [B, L, N, D] it's reshaped internally to merge temporal frames.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.concat = concat
        if concat:
            self.d_k = d_model // n_heads
            assert (
                d_model % n_heads == 0
            ), "d_model must be divisible by n_heads when concat=True"
        else:
            self.d_k = d_model
        self.W = nn.Linear(d_model, self.d_k * n_heads, bias=False)
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.d_k, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        if not concat:
            self.out_proj = nn.Linear(self.d_k, d_model)

    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            B_orig, L, N, D = x.shape
            x = x.view(B_orig * L, N, D)
            B = B_orig * L
            process_4d = True
        else:
            B, N, D = x.shape
            B_orig, L = B, 1
            process_4d = False
        h = self.W(x).view(B, N, self.n_heads, self.d_k)
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)
        edge_h = torch.cat([h_i, h_j], dim=-1)
        edge_e = torch.einsum("bijhd,hd->bijh", edge_h, self.a.squeeze(-1))
        edge_e = self.leakyrelu(edge_e)
        if adj_matrix is not None:
            if adj_matrix.dim() == 2:
                adj_matrix = adj_matrix.unsqueeze(0).expand(B, -1, -1)
            elif adj_matrix.dim() == 3 and process_4d:
                adj_matrix = adj_matrix.unsqueeze(1).expand(-1, L, -1, -1).contiguous().view(B, N, N)
            if adj_matrix.size(0) != B or adj_matrix.size(1) != N or adj_matrix.size(2) != N:
                adj_matrix = adj_matrix[:B, :N, :N]
                if adj_matrix.size(0) < B:
                    adj_matrix = adj_matrix.expand(B, -1, -1)
                if adj_matrix.size(1) < N or adj_matrix.size(2) < N:
                    adj_matrix = F.pad(
                        adj_matrix,
                        (0, max(0, N - adj_matrix.size(2)), 0, max(0, N - adj_matrix.size(1))),
                        value=0,
                    )
                    adj_matrix = adj_matrix[:, :N, :N]
            edge_e = edge_e.masked_fill(adj_matrix.unsqueeze(-1) == 0, -1e9)
        if attn_mask is not None:
            edge_e = edge_e.masked_fill(attn_mask.unsqueeze(-1) == 0, -1e9)
        attention = torch.softmax(edge_e, dim=2)
        attention = self.dropout(attention)
        h_prime = torch.einsum("bijh,bjhd->bihd", attention, h)
        if self.concat:
            output = h_prime.transpose(1, 2).contiguous().view(B, N, -1)
        else:
            output = h_prime.mean(dim=1)
            output = self.out_proj(output)
        if process_4d:
            output = output.view(B_orig, L, N, -1)
            attention = attention.mean(dim=-1).view(B_orig, L, N, N)
        else:
            attention = attention.mean(dim=-1)
        return output, attention

__all__ = ["GraphAttentionLayer"]
