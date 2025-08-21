"""Graph construction utilities split from graph_attention."""
from __future__ import annotations
import torch


def construct_correlation_graph(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Construct feature correlation adjacency matrix.

    Produces adjacency over feature dimension (D).
    """
    B, L, D = x.shape
    x_t = x.transpose(1, 2)
    x_c = x_t - x_t.mean(dim=2, keepdim=True)
    corr = torch.bmm(x_c, x_c.transpose(1, 2)) / (L - 1)
    std = torch.sqrt(torch.diagonal(corr, dim1=1, dim2=2)).unsqueeze(-1)
    corr = corr / (std * std.transpose(1, 2) + 1e-8)
    adj = (corr.abs() > threshold).float()
    identity = torch.eye(D, device=x.device).unsqueeze(0)
    return adj * (1 - identity)


def construct_temporal_correlation_graph(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Construct temporal correlation adjacency matrix.

    Produces adjacency over temporal length (L).
    """
    B, L, D = x.shape
    x_centered = x - x.mean(dim=2, keepdim=True)
    corr = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (D - 1)
    std = torch.sqrt(torch.diagonal(corr, dim1=1, dim2=2)).unsqueeze(-1)
    corr = corr / (std * std.transpose(1, 2) + 1e-8)
    adj = (corr.abs() > threshold).float()
    identity = torch.eye(L, device=x.device).unsqueeze(0)
    return adj * (1 - identity)


def construct_knn_graph(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Construct kNN adjacency matrix over node dimension.

    x: [B, N, D]
    """
    B, N, D = x.shape
    x_norm = (x ** 2).sum(dim=-1, keepdim=True)
    distances = x_norm + x_norm.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
    _, indices = torch.topk(distances, k + 1, dim=-1, largest=False)
    indices = indices[:, :, 1:]
    adj_dir = torch.zeros(B, N, N, device=x.device)
    batch_idx = torch.arange(B).view(-1, 1, 1).expand(-1, N, k)
    node_idx = torch.arange(N).view(1, -1, 1).expand(B, -1, k)
    adj_dir[batch_idx, node_idx, indices] = 1.0
    return adj_dir * adj_dir.transpose(1, 2)

__all__ = [
    "construct_correlation_graph",
    "construct_temporal_correlation_graph",
    "construct_knn_graph",
]
