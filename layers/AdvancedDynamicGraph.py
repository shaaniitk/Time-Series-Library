"""
Advanced Dynamic Cross-Covariate Graph Learner.

Extends the dense O(C^2) DynamicGraphLearner with:
  1. Sparse graph structure learning (top-k sparsification per node)
  2. Multi-hop GNN message passing (stackable layers with residual)
  3. Temporal graph evolution (GRU-based adjacency perturbation)
  4. Edge-aware aggregation (learnable edge features)

All features are composable and gated by constructor flags.
API is a drop-in replacement for DynamicGraphLearner.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGraphStructureLearner(nn.Module):
    """Learns a sparse adjacency via bilinear node embeddings + top-k."""

    def __init__(self, d_model: int, top_k: int, temperature: float = 1.0):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        return_structure: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        Args:
            x: [N, C, d] where N = B*T (flattened batch-time)
        Returns:
            adj: [N, C, C] sparse adjacency (soft weights, zero for non-top-k)
        """
        Q = self.query_proj(x)  # [N, C, d]
        K = self.key_proj(x)    # [N, C, d]
        # Bilinear similarity
        logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [N, C, C]

        C = x.shape[1]
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0.")
        if self.top_k == 0 or self.top_k >= C:
            mask = None
            logits_masked = logits
        else:
            k = min(self.top_k, C)
            _, topk_idx = torch.topk(logits, k, dim=-1)  # [N, C, k]
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, topk_idx, True)
            logits_masked = logits.masked_fill(~mask, float('-inf'))
        adj = F.softmax(logits_masked / self.temperature, dim=-1)  # [N, C, C]
        adj = adj.nan_to_num(0.0)
        if return_structure:
            return adj, logits, mask
        return adj


class GraphMessagePassingLayer(nn.Module):
    """Single GNN layer: sparse attention aggregation + FFN + residual."""

    def __init__(self, d_model: int, dropout: float = 0.1, use_edge_features: bool = False):
        super().__init__()
        self.use_edge_features = use_edge_features

        if use_edge_features:
            self.edge_mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.edge_gate = nn.Linear(d_model, 1)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   [N, C, d]
            adj: [N, C, C] (sparse adjacency weights)
        Returns:
            out: [N, C, d]
        """
        if self.use_edge_features:
            # Build pairwise edge features
            N, C, d = x.shape
            xi = x.unsqueeze(2).expand(N, C, C, d)   # [N, C, C, d] - source
            xj = x.unsqueeze(1).expand(N, C, C, d)   # [N, C, C, d] - target
            edge_feat = self.edge_mlp(torch.cat([xi, xj], dim=-1))  # [N, C, C, d]
            edge_weight = torch.sigmoid(self.edge_gate(edge_feat)).squeeze(-1)  # [N, C, C]
            # Modulate adjacency with edge importance
            adj_eff = adj * edge_weight
            # Normalize rows
            row_sum = adj_eff.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            adj_eff = adj_eff / row_sum
        else:
            adj_eff = adj

        # Message passing: weighted aggregation of neighbor features
        agg = torch.bmm(adj_eff, x)  # [N, C, d]
        x = self.norm1(x + self.dropout(agg))

        # FFN
        x = self.norm2(x + self.ffn(x))
        return x


class TemporalGraphEvolution(nn.Module):
    """
    Learns time-varying adjacency perturbations via a lightweight GRU.
    Base adjacency captures static relationships; GRU captures temporal shifts.
    """

    def __init__(self, d_model: int, num_nodes: int, rank: Optional[int] = None):
        super().__init__()
        # Compress node features to a graph-level summary per timestep
        self.node_compress = nn.Linear(d_model, 1)
        self.num_nodes = num_nodes
        self.rank = max(1, min(num_nodes, rank or min(16, num_nodes)))
        # GRU now evolves a low-rank graph state instead of a dense C^2 state.
        self.gru = nn.GRU(num_nodes, 2 * self.rank, batch_first=True)
        self.src_projection = nn.Linear(2 * self.rank, num_nodes * self.rank)
        self.dst_projection = nn.Linear(2 * self.rank, num_nodes * self.rank)
        # Learnable gate controlling perturbation strength
        alpha_init = torch.logit(torch.tensor(0.1))
        self.alpha_logit = nn.Parameter(alpha_init)

    def forward(
        self,
        x_4d: torch.Tensor,
        base_logits: torch.Tensor,
        structure_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x_4d:    [B, T, C, d] original 4D input
            base_logits: [B*T, C, C] base logits from structure learner
            structure_mask: [B*T, C, C] boolean support mask, or None for dense
        Returns:
            evolved_adj: [B*T, C, C] time-varying adjacency
        """
        B, T, C, d = x_4d.shape
        # Graph summary per timestep: [B, T, C]
        summary = self.node_compress(x_4d).squeeze(-1)  # [B, T, C]
        # GRU produces per-timestep perturbation
        state, _ = self.gru(summary)  # [B, T, 2*rank]
        state = torch.tanh(state).reshape(B * T, 2 * self.rank)
        src = self.src_projection(state).reshape(B * T, C, self.rank)
        dst = self.dst_projection(state).reshape(B * T, C, self.rank)
        delta_logits = torch.bmm(src, dst.transpose(1, 2))
        delta_logits = torch.tanh(delta_logits) * torch.sigmoid(self.alpha_logit)
        evolved_logits = base_logits + delta_logits
        if structure_mask is not None:
            evolved_logits = evolved_logits.masked_fill(~structure_mask, float('-inf'))
        evolved = F.softmax(evolved_logits, dim=-1)
        evolved = evolved.nan_to_num(0.0)
        return evolved


class AdvancedDynamicGraphLearner(nn.Module):
    """
    Drop-in replacement for DynamicGraphLearner with configurable advanced features.

    Args:
        d_model:      Feature dimension per node
        n_heads:       Number of attention heads (kept for API compat, used in output reshape)
        dropout:       Dropout rate
        output_attention: Whether to support returning attention weights
        top_k:         Max neighbors per node for sparse graph (0 = dense)
        num_layers:    Number of GNN message passing layers
        temporal_evolution: Enable GRU-based temporal adjacency evolution
        edge_features:  Enable learnable edge features in message passing
        num_nodes:      Number of covariates (required if temporal_evolution=True)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        output_attention: bool = False,
        top_k: int = 10,
        num_layers: int = 2,
        temporal_evolution: bool = False,
        edge_features: bool = False,
        num_nodes: int = 0,
    ):
        super().__init__()
        self.output_attention = output_attention
        self.n_heads = n_heads
        self.top_k = top_k
        self.num_layers = num_layers
        self.temporal_evolution = temporal_evolution
        self.edge_features = edge_features
        self.max_edge_feature_elements = 2_000_000

        # 1. Sparse structure learner
        self.structure_learner = SparseGraphStructureLearner(d_model, top_k)

        # 2. Multi-hop GNN layers with DenseNet-style skip connections
        self.gnn_layers = nn.ModuleList([
            GraphMessagePassingLayer(d_model, dropout, use_edge_features=(edge_features and i == 0))
            for i in range(num_layers)
        ])
        # Skip connection projections (input + each layer output → final)
        self.skip_proj = nn.Linear(d_model * (num_layers + 1), d_model)
        self.final_norm = nn.LayerNorm(d_model)

        # 3. Temporal evolution (optional)
        self.temporal_evolver = None
        if temporal_evolution and num_nodes > 0:
            self.temporal_evolver = TemporalGraphEvolution(d_model, num_nodes)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_attention: Optional[bool] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        API-compatible with DynamicGraphLearner.

        x shapes:
            Temporal: [B, T, C, d_model]
            Static:   [B, C, d_model]
        """
        return_attention = self.output_attention if return_attention is None else return_attention

        if x.ndim == 4:
            return self._forward_temporal(x, return_attention)
        elif x.ndim == 3:
            return self._forward_static(x, return_attention)
        else:
            raise ValueError(f"AdvancedDynamicGraphLearner expects rank-3 or rank-4 input, got {x.ndim}.")

    def _forward_temporal(
        self, x: torch.Tensor, return_attention: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C, d = x.shape
        x_4d = x  # keep for temporal evolution
        x_flat = x.reshape(B * T, C, d)

        if self.edge_features:
            edge_elements = B * T * C * C * d
            if edge_elements > self.max_edge_feature_elements:
                raise ValueError(
                    "edge_features would materialize an oversized dense edge tensor; "
                    "disable edge_features or reduce batch/time/node dimensions."
                )

        # Learn sparse adjacency
        adj, base_logits, structure_mask = self.structure_learner(x_flat, return_structure=True)  # [B*T, C, C]

        # Temporal evolution of adjacency
        if self.temporal_evolver is not None:
            adj = self.temporal_evolver(x_4d, base_logits, structure_mask)

        # Multi-hop message passing with skip connections
        skip_outputs = [x_flat]  # input as first skip
        h = x_flat
        for layer in self.gnn_layers:
            h = layer(h, adj)
            skip_outputs.append(h)

        # DenseNet aggregation: concat all layer outputs → project
        combined = torch.cat(skip_outputs, dim=-1)  # [B*T, C, d*(L+1)]
        out_flat = self.skip_proj(combined)          # [B*T, C, d]
        out_flat = self.final_norm(x_flat + self.dropout(out_flat))  # residual

        out = out_flat.reshape(B, T, C, d)

        if return_attention:
            # Return adjacency as "attention weights" in expected shape [B, T, n_heads, C, C]
            # Broadcast single-head adjacency across n_heads for API compat
            adj_5d = adj.reshape(B, T, C, C).unsqueeze(2).expand(B, T, self.n_heads, C, C)
            return out, adj_5d
        return out

    def _forward_static(
        self, x: torch.Tensor, return_attention: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, C, d = x.shape

        adj = self.structure_learner(x)  # [B, C, C]

        skip_outputs = [x]
        h = x
        for layer in self.gnn_layers:
            h = layer(h, adj)
            skip_outputs.append(h)

        combined = torch.cat(skip_outputs, dim=-1)
        out = self.skip_proj(combined)
        out = self.final_norm(x + self.dropout(out))

        if return_attention:
            adj_4d = adj.unsqueeze(1).expand(B, self.n_heads, C, C)
            return out, adj_4d
        return out
