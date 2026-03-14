from typing import Optional

import torch
import torch.nn as nn


def build_causal_mask(seq_len: int, device, dtype):
    return torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype), 1)


class MultiScaleLagAttention(nn.Module):
    def __init__(self, d_model, n_heads, lag_scales, dropout=0.0):
        super(MultiScaleLagAttention, self).__init__()
        if not isinstance(lag_scales, (list, tuple)) or len(lag_scales) == 0:
            raise ValueError("tft_lag_scales must be a non-empty list/tuple of positive integers.")
        normalized_lags = []
        for lag in lag_scales:
            if not isinstance(lag, int) or lag <= 0:
                raise ValueError("tft_lag_scales must contain positive integers only.")
            normalized_lags.append(lag)
        if len(set(normalized_lags)) != len(normalized_lags):
            raise ValueError("tft_lag_scales contains duplicated lag values.")

        self.lag_scales = tuple(normalized_lags)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in self.lag_scales
        ])
        self.scale_logits = nn.Parameter(torch.zeros(len(self.lag_scales)))
        self.out_projection = nn.Linear(d_model, d_model)
        self.out_dropout = nn.Dropout(dropout)

    def _shift_sequence(self, x, lag: int):
        shifted = torch.zeros_like(x)
        if lag < x.shape[1]:
            shifted[:, lag:, :] = x[:, :-lag, :]
        return shifted

    def forward(self, x, return_attention: bool = False):
        if x.ndim != 3:
            raise ValueError(f"MultiScaleLagAttention expects rank-3 [B,T,D] input, got shape {tuple(x.shape)}.")
        if not torch.isfinite(x).all():
            raise ValueError("Lag attention input contains NaN/Inf values.")

        seq_len = x.shape[1]
        attn_mask = build_causal_mask(seq_len, x.device, x.dtype)
        branch_outputs = []
        branch_weights = []
        for lag, attention_layer in zip(self.lag_scales, self.attention_layers):
            shifted = self._shift_sequence(x, lag)
            attn_out, attn_prob = attention_layer(
                x,
                shifted,
                shifted,
                need_weights=return_attention,
                attn_mask=attn_mask,
                average_attn_weights=False,
            )
            if not torch.isfinite(attn_out).all():
                raise ValueError("Lag attention output contains NaN/Inf values.")
            branch_outputs.append(attn_out)
            if return_attention:
                if attn_prob is None or not torch.isfinite(attn_prob).all():
                    raise ValueError("Lag attention weights contain NaN/Inf values.")
                branch_weights.append(attn_prob)

        scale_weights = torch.softmax(self.scale_logits, dim=0)
        fused = sum(weight * branch for weight, branch in zip(scale_weights, branch_outputs))
        fused = self.out_projection(self.out_dropout(fused))

        if return_attention:
            return fused, {
                'lag_attention': torch.stack(branch_weights, dim=-1),
                'lag_scale_weights': scale_weights.detach(),
                'lag_scales': self.lag_scales,
            }
        return fused


class HigherOrderInteractionBlock(nn.Module):
    def __init__(self, d_model, interaction_order=2, interaction_rank=None, dropout=0.0):
        super(HigherOrderInteractionBlock, self).__init__()
        if interaction_order not in (2, 3):
            raise ValueError("tft_interaction_order must be either 2 or 3.")
        if interaction_rank is None:
            interaction_rank = max(4, d_model // 4)
        if not isinstance(interaction_rank, int) or interaction_rank <= 0:
            raise ValueError("tft_interaction_rank must be a positive integer.")

        self.interaction_order = interaction_order
        self.interaction_rank = interaction_rank
        self.left_projection = nn.Linear(d_model, interaction_rank)
        self.right_projection = nn.Linear(d_model, interaction_rank)
        self.pair_projection = nn.Linear(interaction_rank, d_model)
        if self.interaction_order == 3:
            self.third_projection = nn.Linear(d_model, interaction_rank)
            self.triple_projection = nn.Linear(interaction_rank, d_model)
        else:
            self.third_projection = None
            self.triple_projection = None
        self.gate_projection = nn.Linear(d_model, self.interaction_order)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, return_payload: bool = False):
        if x.ndim != 3:
            raise ValueError(f"HigherOrderInteractionBlock expects rank-3 [B,T,D] input, got shape {tuple(x.shape)}.")
        if not torch.isfinite(x).all():
            raise ValueError("Higher-order interaction input contains NaN/Inf values.")

        left = self.left_projection(x)
        right = self.right_projection(x)
        pair_term = self.pair_projection((left * right) / (self.interaction_rank ** 0.5))
        interaction_terms = [pair_term]

        if self.interaction_order == 3:
            third = self.third_projection(x)
            triple_term = self.triple_projection((left * right * third) / self.interaction_rank)
            interaction_terms.append(triple_term)

        gates = torch.softmax(self.gate_projection(x), dim=-1)
        interaction_stack = torch.stack(interaction_terms, dim=-2)
        contribution = torch.sum(gates.unsqueeze(-1) * interaction_stack, dim=-2)
        out = self.layer_norm(x + self.out_projection(self.dropout(contribution)))
        if not torch.isfinite(out).all():
            raise ValueError("Higher-order interaction output contains NaN/Inf values.")

        if return_payload:
            return out, {
                'interaction_contribution': contribution.detach(),
                'interaction_gates': gates.detach(),
            }
        return out


class RegimeAwareSparseMoE(nn.Module):
    def __init__(
        self,
        d_model,
        num_experts=4,
        top_k=2,
        num_regimes=4,
        hidden_size=None,
        dropout=0.0,
        noise_epsilon=1e-2,
    ):
        super(RegimeAwareSparseMoE, self).__init__()
        if num_experts <= 0:
            raise ValueError("tft_num_moe_experts must be positive.")
        if num_regimes <= 0:
            raise ValueError("tft_num_regimes must be positive.")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("tft_moe_top_k must be in [1, tft_num_moe_experts].")
        hidden_size = d_model if hidden_size is None else hidden_size

        self.num_experts = num_experts
        self.top_k = top_k
        self.num_regimes = num_regimes
        self.noise_epsilon = noise_epsilon
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.regime_detector = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_regimes),
        )
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.noise = nn.Linear(d_model, num_experts, bias=False)
        self.regime_expert_bias = nn.Parameter(torch.zeros(num_regimes, num_experts))
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, d_model),
            )
            for _ in range(num_experts)
        ])
        self.out_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def cv_squared(self, x):
        eps = 1e-10
        if x.numel() <= 1:
            return x.new_tensor(0.0)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _compute_sparse_routing(self, x, regime_probs):
        logits = self.gate(x)
        if self.training:
            raw_noise = self.noise(x)
            noise_std = self.softplus(raw_noise) + self.noise_epsilon
            logits = logits + torch.randn_like(logits) * noise_std

        logits = logits + torch.einsum('br,re->be', regime_probs, self.regime_expert_bias).unsqueeze(1)
        dense_probs = self.softmax(logits)
        top_values, top_indices = torch.topk(dense_probs, self.top_k, dim=-1)
        sparse_probs = torch.zeros_like(dense_probs)
        sparse_probs.scatter_(-1, top_indices, top_values)
        sparse_probs = sparse_probs / sparse_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        importance = sparse_probs.sum(dim=(0, 1))
        aux_loss = self.cv_squared(importance)
        return sparse_probs, aux_loss

    def forward(self, x, context: Optional[torch.Tensor] = None, return_payload: bool = False):
        if x.ndim != 3:
            raise ValueError(f"RegimeAwareSparseMoE expects rank-3 [B,T,D] input, got shape {tuple(x.shape)}.")
        if not torch.isfinite(x).all():
            raise ValueError("MoE input contains NaN/Inf values.")

        pooled_context = x.mean(dim=1) if context is None else context
        if pooled_context.ndim != 2:
            raise ValueError(f"MoE context must be rank-2 [B,D], got shape {tuple(pooled_context.shape)}.")
        regime_probs = self.softmax(self.regime_detector(pooled_context))
        routing, aux_loss = self._compute_sparse_routing(x, regime_probs)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        mixed = torch.sum(routing.unsqueeze(-1) * expert_outputs, dim=-2)
        out = self.layer_norm(x + self.out_dropout(mixed))
        if not torch.isfinite(out).all():
            raise ValueError("MoE output contains NaN/Inf values.")

        if return_payload:
            return out, aux_loss, {
                'expert_routing': routing.detach(),
                'regime_probabilities': regime_probs.detach(),
            }
        return out, aux_loss
