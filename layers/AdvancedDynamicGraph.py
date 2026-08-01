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

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGraphStructureLearner(nn.Module):
    """Learns sparse adjacency with explicit self-edge and head semantics."""

    VALID_SELF_EDGE_POLICIES = frozenset({"required", "allowed", "excluded"})
    VALID_HEAD_MODES = frozenset({"single", "true_multihead"})

    def __init__(
        self,
        d_model: int,
        top_k: int,
        temperature: float = 1.0,
        density: Optional[float] = None,
        self_edge_policy: str = "allowed",
        n_heads: int = 1,
        head_mode: str = "single",
    ):
        super().__init__()
        if top_k < 0:
            raise ValueError("top_k must be >= 0.")
        if temperature <= 0 or not math.isfinite(float(temperature)):
            raise ValueError("temperature must be a positive finite value.")
        if density is not None:
            density = float(density)
            if density <= 0.0 or density > 1.0 or not math.isfinite(density):
                raise ValueError("density must be in (0, 1] when provided.")
        if self_edge_policy not in self.VALID_SELF_EDGE_POLICIES:
            raise ValueError(
                "self_edge_policy must be one of "
                f"{sorted(self.VALID_SELF_EDGE_POLICIES)}, got {self_edge_policy!r}."
            )
        if head_mode not in self.VALID_HEAD_MODES:
            raise ValueError(
                "head_mode must be one of "
                f"{sorted(self.VALID_HEAD_MODES)}, got {head_mode!r}."
            )
        if not isinstance(n_heads, int) or n_heads < 1:
            raise ValueError("n_heads must be an integer >= 1.")

        self.top_k = int(top_k)
        self.density = density
        self.temperature = float(temperature)
        self.self_edge_policy = self_edge_policy
        self.head_mode = head_mode
        self.n_heads = 1 if head_mode == "single" else int(n_heads)

        out_dim = d_model * self.n_heads
        self.query_proj = nn.Linear(d_model, out_dim, bias=False)
        self.key_proj = nn.Linear(d_model, out_dim, bias=False)
        self.scale = d_model ** -0.5

    def _resolve_k(self, num_nodes: int) -> int:
        if self.self_edge_policy in {"required", "excluded"}:
            candidate_count = num_nodes - 1
        else:
            candidate_count = num_nodes
        if candidate_count < 1:
            raise ValueError("Graph mixing requires at least 2 nodes for SR08 policies.")

        if self.density is not None:
            k = int(math.ceil(self.density * candidate_count))
        else:
            k = self.top_k

        if k == 0:
            return candidate_count
        if k > candidate_count:
            raise ValueError(
                f"Resolved graph sparsity k={k} exceeds available candidates {candidate_count} "
                f"for self_edge_policy={self.self_edge_policy!r}."
            )
        return k

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
        if x.ndim != 3:
            raise ValueError(f"SparseGraphStructureLearner expects [N,C,d], got {tuple(x.shape)}.")

        N, C, d_model = x.shape
        Q = self.query_proj(x).reshape(N, C, self.n_heads, d_model).permute(0, 2, 1, 3)
        K = self.key_proj(x).reshape(N, C, self.n_heads, d_model).permute(0, 2, 1, 3)
        logits = torch.matmul(Q, K.transpose(-1, -2)) * self.scale  # [N,H,C,C]

        k = self._resolve_k(C)
        diag = torch.eye(C, device=x.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        allowed = torch.ones_like(logits, dtype=torch.bool)
        if self.self_edge_policy == "excluded":
            allowed = allowed & (~diag)

        support = None
        if k < C:
            selection_logits = logits
            if self.self_edge_policy in {"required", "excluded"}:
                selection_logits = selection_logits.masked_fill(diag, float("-inf"))
            _, topk_idx = torch.topk(selection_logits, k, dim=-1)
            support = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, topk_idx, True)
            if self.self_edge_policy == "required":
                support = support | diag
            elif self.self_edge_policy == "allowed":
                pass
            else:
                support = support & (~diag)
        else:
            support = allowed
            if self.self_edge_policy == "required":
                support = support | diag

        support = support & allowed
        logits_masked = logits.masked_fill(~support, float("-inf"))
        adj = F.softmax(logits_masked / self.temperature, dim=-1)  # [N,H,C,C]
        adj = adj.nan_to_num(0.0)

        if self.n_heads == 1:
            adj_out = adj[:, 0]
            logits_out = logits[:, 0]
            support_out = support[:, 0]
        else:
            adj_out = adj
            logits_out = logits
            support_out = support

        if return_structure:
            return adj_out, logits_out, support_out
        return adj_out


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
        density: Optional[float] = None,
        num_layers: int = 2,
        temporal_evolution: bool = False,
        edge_features: bool = False,
        num_nodes: int = 0,
        self_edge_policy: str = "allowed",
        head_mode: str = "single",
        temperature: float = 1.0,
        entropy_regularization: float = 0.0,
        support_stability_regularization: float = 0.0,
        residual_strength_init: float = 1e-3,
        graph_scope: str = "observed_and_known",
    ):
        super().__init__()
        self.output_attention = output_attention
        self.n_heads = n_heads
        self.top_k = top_k
        self.density = density
        self.num_layers = num_layers
        self.temporal_evolution = temporal_evolution
        self.edge_features = edge_features
        self.self_edge_policy = self_edge_policy
        self.head_mode = head_mode
        self.temperature = float(temperature)
        self.entropy_regularization = float(entropy_regularization)
        self.support_stability_regularization = float(support_stability_regularization)
        self.graph_scope = graph_scope
        self.max_edge_feature_elements = 2_000_000
        if residual_strength_init < 0.0 or not math.isfinite(float(residual_strength_init)):
            raise ValueError("residual_strength_init must be a finite value >= 0.")
        self.residual_strength = nn.Parameter(
            torch.tensor(float(residual_strength_init))
        )
        self.last_graph_metadata = None

        # 1. Sparse structure learner
        self.structure_learner = SparseGraphStructureLearner(
            d_model=d_model,
            top_k=top_k,
            temperature=temperature,
            density=density,
            self_edge_policy=self_edge_policy,
            n_heads=n_heads,
            head_mode=head_mode,
        )

        # 2. Multi-hop GNN layers with DenseNet-style skip connections
        self.gnn_layers = nn.ModuleList([
            GraphMessagePassingLayer(d_model, dropout, use_edge_features=(edge_features and i == 0))
            for i in range(num_layers)
        ])
        # Skip connection projections (input + each layer output → final)
        self.skip_proj = nn.Linear(d_model * (num_layers + 1), d_model)
        self.branch_norm = nn.LayerNorm(d_model)

        # 3. Temporal evolution (optional)
        self.temporal_evolver = None
        if temporal_evolution and num_nodes > 0:
            self.temporal_evolver = TemporalGraphEvolution(d_model, num_nodes)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _ensure_head_dim(adj: torch.Tensor) -> torch.Tensor:
        if adj.ndim == 3:
            return adj.unsqueeze(1)
        if adj.ndim == 4:
            return adj
        raise ValueError(f"Expected adjacency rank 3/4, got {adj.ndim}.")

    def _build_metadata(
        self,
        adj_heads: torch.Tensor,
        support_heads: torch.Tensor,
        batch_size: int,
        seq_len: Optional[int] = None,
    ):
        eps = torch.finfo(adj_heads.dtype).tiny
        entropy = -(adj_heads.clamp_min(eps) * adj_heads.clamp_min(eps).log()).sum(dim=-1)
        support_frequency = support_heads.to(dtype=adj_heads.dtype).mean()
        self_edge_mass = torch.diagonal(adj_heads, dim1=-2, dim2=-1).mean()

        if seq_len is not None and seq_len > 1:
            support_bt = support_heads.reshape(batch_size, seq_len, *support_heads.shape[1:])
            support_turnover = (
                support_bt[:, 1:] ^ support_bt[:, :-1]
            ).to(dtype=adj_heads.dtype).mean()
        else:
            support_turnover = adj_heads.new_tensor(0.0)

        return {
            "graph_scope": self.graph_scope,
            "head_mode": self.head_mode,
            "num_reported_heads": int(adj_heads.shape[1]),
            "self_edge_policy": self.self_edge_policy,
            "temperature": float(self.temperature),
            "entropy_regularization": float(self.entropy_regularization),
            "support_stability_regularization": float(self.support_stability_regularization),
            "adjacency_entropy": entropy.mean().detach(),
            "selected_support_frequency": support_frequency.detach(),
            "support_turnover": support_turnover.detach(),
            "self_edge_mass": self_edge_mass.detach(),
            "residual_strength": self.residual_strength.detach().clone(),
        }

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
        adj, base_logits, structure_mask = self.structure_learner(
            x_flat,
            return_structure=True,
        )
        adj_heads = self._ensure_head_dim(adj)
        support_heads = self._ensure_head_dim(structure_mask)

        # Temporal evolution of adjacency
        if self.temporal_evolver is not None:
            # Temporal evolution is single-adjacency in SR08; keep head_mode single.
            if adj_heads.shape[1] != 1:
                raise ValueError(
                    "temporal_evolution currently supports head_mode='single' only."
                )
            evolved = self.temporal_evolver(x_4d, base_logits, structure_mask)
            adj_heads = evolved.unsqueeze(1)
            support_heads = structure_mask.unsqueeze(1)

        # Aggregate head-specific adjacency into one message matrix.
        adj_for_message = adj_heads.mean(dim=1)

        # Multi-hop message passing with skip connections
        skip_outputs = [x_flat]  # input as first skip
        h = x_flat
        for layer in self.gnn_layers:
            h = layer(h, adj_for_message)
            skip_outputs.append(h)

        # DenseNet aggregation: concat all layer outputs → project
        combined = torch.cat(skip_outputs, dim=-1)  # [B*T, C, d*(L+1)]
        branch = self.branch_norm(self.skip_proj(combined))
        out_flat = x_flat + self.dropout(self.residual_strength * branch)

        out = out_flat.reshape(B, T, C, d)
        self.last_graph_metadata = self._build_metadata(
            adj_heads,
            support_heads,
            batch_size=B,
            seq_len=T,
        )

        if return_attention:
            adj_5d = adj_heads.reshape(B, T, adj_heads.shape[1], C, C)
            return out, adj_5d
        return out

    def _forward_static(
        self, x: torch.Tensor, return_attention: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, C, d = x.shape

        adj = self.structure_learner(x)
        adj_heads = self._ensure_head_dim(adj)
        adj_for_message = adj_heads.mean(dim=1)

        skip_outputs = [x]
        h = x
        for layer in self.gnn_layers:
            h = layer(h, adj_for_message)
            skip_outputs.append(h)

        combined = torch.cat(skip_outputs, dim=-1)
        branch = self.branch_norm(self.skip_proj(combined))
        out = x + self.dropout(self.residual_strength * branch)
        self.last_graph_metadata = self._build_metadata(
            adj_heads,
            adj_heads > 0.0,
            batch_size=B,
            seq_len=None,
        )

        if return_attention:
            adj_4d = adj_heads
            return out, adj_4d
        return out
