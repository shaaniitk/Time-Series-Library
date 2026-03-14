import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_causal_mask(seq_len: int, device, dtype):
    return torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype), 1)


def _resolve_positions(length: int, positions, device):
    if positions is None:
        return torch.arange(length, device=device, dtype=torch.float32)
    return positions.to(device=device, dtype=torch.float32)


def _rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(start_dim=-2)


def apply_rotary_embedding(query, key, query_positions=None, key_positions=None, base=10000.0):
    if query.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")

    device = query.device
    query_positions = _resolve_positions(query.shape[-2], query_positions, device)
    key_positions = _resolve_positions(key.shape[-2], key_positions, device)
    half_dim = query.shape[-1] // 2
    index = torch.arange(half_dim, device=device, dtype=torch.float32)
    inv_freq = base ** (-index / max(1, half_dim))

    def _apply(x, positions):
        angles = torch.einsum('t,d->td', positions, inv_freq)
        sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1).to(dtype=x.dtype)
        cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1).to(dtype=x.dtype)
        return x * cos.unsqueeze(0).unsqueeze(0) + _rotate_half(x) * sin.unsqueeze(0).unsqueeze(0)

    return _apply(query, query_positions), _apply(key, key_positions)


def _get_alibi_slopes(num_heads: int, device, dtype):
    def _power_of_two_slopes(power_of_two: int):
        start = 2 ** (-(2 ** -(math.log2(power_of_two) - 3)))
        ratio = start
        return [start * (ratio ** idx) for idx in range(power_of_two)]

    if num_heads <= 0:
        raise ValueError("num_heads must be positive for ALiBi.")
    if math.log2(num_heads).is_integer():
        slopes = _power_of_two_slopes(num_heads)
    else:
        closest_power_of_two = 2 ** math.floor(math.log2(num_heads))
        slopes = _power_of_two_slopes(closest_power_of_two)
        extra = _power_of_two_slopes(2 * closest_power_of_two)
        slopes.extend(extra[0::2][: num_heads - closest_power_of_two])
    return torch.tensor(slopes, device=device, dtype=dtype)


def build_alibi_bias(num_heads, query_len, key_len, device, dtype, query_positions=None, key_positions=None, scale=1.0):
    query_positions = _resolve_positions(query_len, query_positions, device)
    key_positions = _resolve_positions(key_len, key_positions, device)
    relative_distance = (query_positions.unsqueeze(-1) - key_positions.unsqueeze(0)).abs()
    slopes = _get_alibi_slopes(num_heads, device, dtype)
    return -(scale * slopes.view(1, num_heads, 1, 1) * relative_distance.view(1, 1, query_len, key_len).to(dtype=dtype))


class PositionalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, position_bias_type='none', rope_base=10000.0, alibi_scale=1.0):
        super(PositionalMultiHeadAttention, self).__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for PositionalMultiHeadAttention.")
        if position_bias_type not in {'none', 'rope', 'alibi'}:
            raise ValueError("position_bias_type must be one of: none, rope, alibi.")

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.position_bias_type = position_bias_type
        self.rope_base = float(rope_base)
        self.alibi_scale = float(alibi_scale)
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.out_projection = nn.Linear(d_model, d_model, bias=False)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(
        self,
        query,
        key,
        value,
        return_attention: bool = False,
        attn_mask=None,
        query_positions=None,
        key_positions=None,
    ):
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError(
                f"PositionalMultiHeadAttention expects rank-3 tensors, got {tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
            )
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        q = self.q_linear(query).view(batch_size, query_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_linear(key).view(batch_size, key_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_linear(value).view(batch_size, key_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        if self.position_bias_type == 'rope':
            q, k = apply_rotary_embedding(q, k, query_positions=query_positions, key_positions=key_positions, base=self.rope_base)

        attention_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.position_bias_type == 'alibi':
            attention_score = attention_score + build_alibi_bias(
                self.n_heads,
                query_len,
                key_len,
                query.device,
                attention_score.dtype,
                query_positions=query_positions,
                key_positions=key_positions,
                scale=self.alibi_scale,
            )

        if not torch.isfinite(attention_score).all():
            raise ValueError("Attention scores contain NaN/Inf values before masking.")
        clamp_limit = 1e4
        if (attention_score.abs() > clamp_limit).any():
            attention_score = attention_score.clamp(min=-clamp_limit, max=clamp_limit)

        if attn_mask is not None:
            if attn_mask.ndim != 2:
                raise ValueError(f"attn_mask must be rank-2 [T,S], got shape {tuple(attn_mask.shape)}.")
            attention_score = attention_score + attn_mask.unsqueeze(0).unsqueeze(0)
        attention_prob = F.softmax(attention_score, dim=-1)
        if not torch.isfinite(attention_prob).all():
            raise ValueError("Attention probabilities contain NaN/Inf values.")

        attention_out = torch.matmul(attention_prob, v)
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous().view(batch_size, query_len, self.n_heads * self.d_head)
        out = self.out_projection(attention_out)
        out = self.out_dropout(out)
        if return_attention:
            return out, attention_prob
        return out


class MultiScaleLagAttention(nn.Module):
    def __init__(self, d_model, n_heads, lag_scales, dropout=0.0, position_bias_type='none', rope_base=10000.0, alibi_scale=1.0):
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
            PositionalMultiHeadAttention(
                d_model,
                n_heads,
                dropout=dropout,
                position_bias_type=position_bias_type,
                rope_base=rope_base,
                alibi_scale=alibi_scale,
            )
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
            if return_attention:
                attn_out, attn_prob = attention_layer(
                    x,
                    shifted,
                    shifted,
                    return_attention=True,
                    attn_mask=attn_mask,
                )
            else:
                attn_out = attention_layer(x, shifted, shifted, attn_mask=attn_mask)
                attn_prob = None
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


class InterpretableCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, position_bias_type='none', rope_base=10000.0, alibi_scale=1.0):
        super(InterpretableCrossAttention, self).__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for InterpretableCrossAttention.")
        if position_bias_type not in {'none', 'rope', 'alibi'}:
            raise ValueError("position_bias_type must be one of: none, rope, alibi.")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.position_bias_type = position_bias_type
        self.rope_base = float(rope_base)
        self.alibi_scale = float(alibi_scale)
        self.q_linear = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_linear = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.v_linear = nn.Linear(d_model, self.d_head, bias=False)
        self.out_projection = nn.Linear(self.d_head, d_model, bias=False)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, query, context, return_attention: bool = False, query_positions=None, key_positions=None):
        if query.ndim != 3 or context.ndim != 3:
            raise ValueError(
                f"InterpretableCrossAttention expects rank-3 query/context, got {tuple(query.shape)} and {tuple(context.shape)}."
            )
        q = self.q_linear(query).view(query.shape[0], query.shape[1], self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_linear(context).view(context.shape[0], context.shape[1], self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_linear(context)

        if self.position_bias_type == 'rope':
            q, k = apply_rotary_embedding(q, k, query_positions=query_positions, key_positions=key_positions, base=self.rope_base)

        attention_score = torch.matmul(q, k.transpose(-2, -1))
        attention_score = attention_score * self.scale
        if self.position_bias_type == 'alibi':
            attention_score = attention_score + build_alibi_bias(
                self.n_heads,
                query.shape[1],
                context.shape[1],
                query.device,
                attention_score.dtype,
                query_positions=query_positions,
                key_positions=key_positions,
                scale=self.alibi_scale,
            )
        if not torch.isfinite(attention_score).all():
            raise ValueError("Cross-attention scores contain NaN/Inf values.")
        attention_prob = torch.softmax(attention_score, dim=-1)
        if not torch.isfinite(attention_prob).all():
            raise ValueError("Cross-attention probabilities contain NaN/Inf values.")

        attention_out = torch.matmul(attention_prob, v.unsqueeze(1))
        attention_out = attention_out.mean(dim=1)
        out = self.out_projection(attention_out)
        out = self.out_dropout(out)
        if return_attention:
            return out, attention_prob
        return out


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
        self.context_to_regime = nn.Linear(d_model, num_regimes, bias=False)
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
        if regime_probs.ndim != 3:
            raise ValueError(f"Expected timestep regime probabilities [B,T,R], got shape {tuple(regime_probs.shape)}.")
        logits = self.gate(x)
        if self.training:
            raw_noise = self.noise(x)
            noise_std = self.softplus(raw_noise) + self.noise_epsilon
            logits = logits + torch.randn_like(logits) * noise_std

        logits = logits + torch.einsum('btr,re->bte', regime_probs, self.regime_expert_bias)
        dense_probs = self.softmax(logits)
        top_values, top_indices = torch.topk(dense_probs, self.top_k, dim=-1)
        sparse_probs = torch.zeros_like(dense_probs)
        sparse_probs.scatter_(-1, top_indices, top_values)
        sparse_probs = sparse_probs / sparse_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        importance = sparse_probs.sum(dim=(0, 1))
        regime_usage = regime_probs.sum(dim=(0, 1))
        aux_loss = self.cv_squared(importance) + 0.1 * self.cv_squared(regime_usage)
        return sparse_probs, aux_loss

    def forward(self, x, context: Optional[torch.Tensor] = None, return_payload: bool = False):
        if x.ndim != 3:
            raise ValueError(f"RegimeAwareSparseMoE expects rank-3 [B,T,D] input, got shape {tuple(x.shape)}.")
        if not torch.isfinite(x).all():
            raise ValueError("MoE input contains NaN/Inf values.")

        regime_logits = self.regime_detector(x)
        if context is not None:
            if context.ndim == 2:
                regime_logits = regime_logits + self.context_to_regime(context).unsqueeze(1)
            elif context.ndim == 3:
                regime_logits = regime_logits + self.context_to_regime(context)
            else:
                raise ValueError(f"MoE context must be rank-2 [B,D] or rank-3 [B,T,D], got shape {tuple(context.shape)}.")
        regime_probs = self.softmax(regime_logits)
        pooled_regime_probs = regime_probs.mean(dim=1)
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
                'regime_probabilities_pooled': pooled_regime_probs.detach(),
            }
        return out, aux_loss
