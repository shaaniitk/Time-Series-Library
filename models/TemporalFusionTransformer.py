import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding, TemporalEmbedding
from layers.DynamicGraph import DynamicGraphLearner
from layers.AdvancedDynamicGraph import AdvancedDynamicGraphLearner
from layers.StandardNorm import Normalize
from layers.TemporalFusion_layers import (
    GatedDilatedTemporalBackbone,
    HigherOrderInteractionBlock,
    HybridTemporalBackbone,
    InterpretableCrossAttention,
    MultiScaleLagAttention,
    PositionalMultiHeadAttention,
    RegimeAwareSparseMoE,
    SpectralBranch,
    TemporalCompression,
    apply_rotary_embedding,
    build_causal_mask,
    build_alibi_bias,
)
from utils.tft_schema import resolve_known_feature_names, resolve_tft_schema
from utils.tft_config import apply_tft_profile
from torch import Tensor
from typing import NamedTuple, Optional
from collections import namedtuple
import warnings
from utils.losses import canonicalize_quantiles

# static: time-independent features
# observed: time features of the past(e.g. predicted targets)
# known: known information about the past and future(i.e. time stamp)
TypePos = namedtuple('TypePos', ['static', 'observed'])


class TFTForecastOutput(NamedTuple):
    point_forecast: torch.Tensor
    point_full: torch.Tensor
    quantile_forecast: Optional[torch.Tensor]
    moe_importance_sum: Optional[torch.Tensor]
    moe_load_sum: Optional[torch.Tensor]
    moe_token_count: Optional[torch.Tensor]

# When you want to use new dataset, please add the index of 'static, observed' columns here.
# 'known' columns needn't be added, because 'known' inputs are automatically judged and provided by the program.
datatype_dict = {
    'ETTh1': TypePos([], [x for x in range(7)]),
    'ETTh2': TypePos([], [x for x in range(7)]),
    'ETTm1': TypePos([], [x for x in range(7)]),
    'ETTm2': TypePos([], [x for x in range(7)]),
}


def get_typepos(configs) -> TypePos:
    resolved_schema = getattr(configs, '_tft_resolved_schema', None)
    if resolved_schema is not None:
        return TypePos(list(resolved_schema.static_positions), list(resolved_schema.observed_positions))

    if configs.data in datatype_dict:
        return datatype_dict[configs.data]

    observed_pos = getattr(configs, 'tft_observed_pos', None)
    static_pos = getattr(configs, 'tft_static_pos', None)
    if observed_pos is None:
        raise KeyError(
            f"Dataset '{configs.data}' is not registered in datatype_dict. "
            "You must provide tft_observed_pos explicitly."
        )

    if static_pos is None:
        static_pos = []

    if not isinstance(observed_pos, (list, tuple)) or len(observed_pos) == 0:
        raise ValueError("tft_observed_pos must be a non-empty list/tuple of feature indices.")
    if not isinstance(static_pos, (list, tuple)):
        raise ValueError("tft_static_pos must be a list/tuple of feature indices.")

    if any((not isinstance(v, int) or v < 0) for v in observed_pos):
        raise ValueError("tft_observed_pos must contain non-negative integers only.")
    if any((not isinstance(v, int) or v < 0) for v in static_pos):
        raise ValueError("tft_static_pos must contain non-negative integers only.")

    if len(set(observed_pos)) != len(observed_pos):
        raise ValueError("tft_observed_pos contains duplicated indices.")
    if len(set(static_pos)) != len(static_pos):
        raise ValueError("tft_static_pos contains duplicated indices.")

    return TypePos(list(static_pos), list(observed_pos))


def get_target_pos(configs) -> list:
    resolved_schema = getattr(configs, '_tft_resolved_schema', None)
    if resolved_schema is not None:
        return list(resolved_schema.target_positions)

    if not hasattr(configs, 'enc_in') or not hasattr(configs, 'c_out'):
        raise KeyError("configs must define enc_in and c_out for TFT target mapping.")

    target_pos = getattr(configs, 'tft_target_pos', None)
    if target_pos is None:
        if configs.c_out == configs.enc_in:
            return [x for x in range(configs.c_out)]
        elif configs.c_out == 1 and configs.enc_in > 1:
            # Default to the last column (OT) for MS (multivariate to univariate) forecasting
            return [configs.enc_in - 1]
        raise KeyError(
            "tft_target_pos is required when c_out != enc_in. "
            "Provide explicit source indices in encoder features for each target channel."
        )

    if not isinstance(target_pos, (list, tuple)) or len(target_pos) != configs.c_out:
        raise ValueError(f"tft_target_pos must be a list/tuple of length c_out={configs.c_out}.")
    if any((not isinstance(v, int) or v < 0 or v >= configs.enc_in) for v in target_pos):
        raise ValueError(f"tft_target_pos values must be integer indices in [0, {configs.enc_in - 1}].")
    if len(set(target_pos)) != len(target_pos):
        raise ValueError("tft_target_pos contains duplicated indices.")
    return list(target_pos)


def get_known_len(embed_type, freq):
    return len(resolve_known_feature_names(embed_type, freq))


class TFTTemporalEmbedding(TemporalEmbedding):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TFTTemporalEmbedding, self).__init__(d_model, embed_type, freq)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        embedding_x = torch.stack([month_x, day_x, weekday_x, hour_x, minute_x], dim=-2) if hasattr(
            self, 'minute_embed') else torch.stack([month_x, day_x, weekday_x, hour_x], dim=-2)
        return embedding_x


class TFTTimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TFTTimeFeatureEmbedding, self).__init__()
        d_inp = get_known_len(embed_type, freq)
        self.embed = nn.ModuleList([nn.Linear(1, d_model, bias=False) for _ in range(d_inp)])

    def forward(self, x):
        return torch.stack([embed(x[:,:,i].unsqueeze(-1)) for i, embed in enumerate(self.embed)], dim=-2)


class TFTCustomKnownEmbedding(nn.Module):
    def __init__(self, d_model, max_channels=512):
        super(TFTCustomKnownEmbedding, self).__init__()
        self.value_projection = nn.Linear(1, d_model, bias=False)
        self.channel_embedding = nn.Embedding(max_channels, d_model)
        self.max_channels = max_channels

    def forward(self, x):
        # x: [B,T,C]
        if x.ndim != 3:
            raise ValueError("Custom known features must be rank-3 [B,T,C].")
        c = x.shape[-1]
        if c > self.max_channels:
            raise ValueError(
                f"Known feature channels ({c}) exceed tft_known_max_channels={self.max_channels}."
            )

        values = self.value_projection(x.unsqueeze(-1))
        channel_ids = torch.arange(c, device=x.device)
        channel_bias = self.channel_embedding(channel_ids).unsqueeze(0).unsqueeze(0)
        return values + channel_bias


class PointwiseContinuousEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(1, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"PointwiseContinuousEmbedding expects [B,T,1], got {tuple(x.shape)}.")
        return self.dropout(self.projection(x))


class TFTEmbedding(nn.Module):
    def __init__(self, configs):
        super(TFTEmbedding, self).__init__()
        self.pred_len = configs.pred_len
        self.allow_custom_known = getattr(configs, 'tft_allow_custom_known', False)
        self.tft_profile = getattr(configs, 'tft_profile', 'extended_safe')
        self.use_canonical_embeddings = self.tft_profile == 'canonical'
        typepos = get_typepos(configs)
        self.static_pos = typepos.static
        self.observed_pos = typepos.observed
        self.static_len = len(self.static_pos)
        self.observed_len = len(self.observed_pos)

        embedding_cls = PointwiseContinuousEmbedding if self.use_canonical_embeddings else DataEmbedding
        if self.use_canonical_embeddings:
            self.static_embedding = nn.ModuleList([embedding_cls(configs.d_model, dropout=configs.dropout) for _ in range(self.static_len)]) \
                if self.static_len else None
            self.observed_embedding = nn.ModuleList([embedding_cls(configs.d_model, dropout=configs.dropout) for _ in range(self.observed_len)])
        else:
            self.static_embedding = nn.ModuleList([embedding_cls(1, configs.d_model, dropout=configs.dropout) for _ in range(self.static_len)]) \
                if self.static_len else None
            self.observed_embedding = nn.ModuleList([embedding_cls(1, configs.d_model, dropout=configs.dropout) for _ in range(self.observed_len)])
        if self.allow_custom_known:
            max_known_channels = getattr(configs, 'tft_known_max_channels', 512)
            self.known_embedding = TFTCustomKnownEmbedding(configs.d_model, max_channels=max_known_channels)
        else:
            self.known_embedding = TFTTemporalEmbedding(configs.d_model, configs.embed, configs.freq) \
                if configs.embed != 'timeF' else TFTTimeFeatureEmbedding(configs.d_model, configs.embed, configs.freq)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, static_values=None):
        if self.static_len and static_values is None:
            raise ValueError("static_values is required when static TFT features are configured.")
        if self.static_len:
            # static_input: [B,C,d_model]
            if static_values.ndim != 2 or static_values.shape[1] != self.static_len:
                raise ValueError(
                    f"static_values must have shape [B,{self.static_len}], got {tuple(static_values.shape)}."
                )
            static_input = torch.stack(
                [
                    (embed(static_values[:, i].view(static_values.shape[0], 1, 1)).squeeze(1)
                     if self.use_canonical_embeddings
                     else embed(static_values[:, i].view(static_values.shape[0], 1, 1), None).squeeze(1))
                    for i, embed in enumerate(self.static_embedding)
                ],
                dim=-2,
            )
        else:
            static_input = None

        # observed_input: [B,T,C,d_model]
        observed_input = torch.stack([
            (embed(x_enc[:, :, self.observed_pos[i]].unsqueeze(-1))
             if self.use_canonical_embeddings
             else embed(x_enc[:, :, self.observed_pos[i]].unsqueeze(-1), None))
            for i, embed in enumerate(self.observed_embedding)
        ], dim=-2)

        x_mark = torch.cat([x_mark_enc, x_mark_dec[:,-self.pred_len:,:]], dim=-2)
        # known_input: [B,T,C,d_model]
        known_input = self.known_embedding(x_mark)

        return static_input, observed_input, known_input


class GLU(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.glu = nn.GLU()

    def forward(self, x):
        a = self.fc1(x)
        b = self.fc2(x)
        return self.glu(torch.cat([a, b], dim=-1))


class SwiGLU(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.silu(self.fc1(x)) * self.fc2(x)


class GateAddNorm(nn.Module):
    def __init__(self, input_size, output_size, use_swiglu=False):
        super(GateAddNorm, self).__init__()
        if use_swiglu:
            self.glu = SwiGLU(input_size, input_size)
        else:
            self.glu = GLU(input_size, input_size)
        self.projection = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x, skip_a):
        x = self.glu(x)
        x = x + skip_a
        return self.layer_norm(self.projection(x))


class GRN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, context_size=None, dropout=0.0, use_swiglu=False):
        super(GRN, self).__init__()
        hidden_size = input_size if hidden_size is None else hidden_size
        self.lin_a = nn.Linear(input_size, hidden_size)
        self.lin_c = nn.Linear(context_size, hidden_size) if context_size is not None else None
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.project_a = nn.Linear(input_size, hidden_size) if hidden_size != input_size else nn.Identity()
        self.gate = GateAddNorm(hidden_size, output_size, use_swiglu=use_swiglu)
        self.use_swiglu = use_swiglu

    def forward(self, a: Tensor, c: Optional[Tensor] = None):
        # a: [B,T,d], c: [B,d]
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        if self.use_swiglu:
            x = F.silu(x)
        else:
            x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        return self.gate(x, self.project_a(a))


class VariableSelectionNetwork(nn.Module):
    def __init__(self, d_model, variable_num, dropout=0.0, use_swiglu=False, cross_variable_mixing=False, n_heads=4, residual_bypass=True, n_selection_heads=1, per_feature_gating=False, low_rank_threshold=64, graph_type='dense', graph_top_k=10, graph_num_layers=2, graph_temporal_evolution=False, graph_edge_features=False, use_context=True):
        super(VariableSelectionNetwork, self).__init__()
        self.per_feature_gating = per_feature_gating
        self.n_selection_heads = n_selection_heads
        self.use_residual_bypass = residual_bypass
        self.use_context = use_context
        if n_selection_heads > 1:
            assert d_model % n_selection_heads == 0, (
                f"d_model ({d_model}) must be divisible by n_selection_heads ({n_selection_heads})."
            )
        if cross_variable_mixing:
            if graph_type == 'dense':
                self.cross_mixing = DynamicGraphLearner(d_model, n_heads, dropout, output_attention=True)
            else:
                self.cross_mixing = AdvancedDynamicGraphLearner(
                    d_model, n_heads, dropout, output_attention=True,
                    top_k=graph_top_k, num_layers=graph_num_layers,
                    temporal_evolution=(graph_type == 'temporal_sparse' or graph_temporal_evolution),
                    edge_features=graph_edge_features,
                    num_nodes=variable_num,
                )
        else:
            self.cross_mixing = None
        # Use low-rank factorization for high-dimensional inputs
        input_dim = d_model * variable_num
        self.use_low_rank = variable_num >= low_rank_threshold
        if self.use_low_rank and not self.per_feature_gating:
            rank = min(d_model, variable_num)
            self.low_rank_down = nn.Linear(input_dim, rank)
            self.low_rank_act = nn.GELU()
            head_input_dim = rank
        else:
            self.low_rank_down = None
            self.low_rank_act = None
            head_input_dim = input_dim
        if per_feature_gating:
            self.head_grns = None
            self.head_context_projections = None
            self.variable_grns = None
            self.feature_gate_grn = GRN(
                d_model * variable_num, variable_num,
                hidden_size=d_model, context_size=d_model,
                dropout=dropout, use_swiglu=use_swiglu,
            )
            self.feature_gate_dropout = nn.Dropout(dropout)
        else:
            # Multi-head selection: each head learns independent variable importance weights
            self.head_grns = nn.ModuleList([
                GRN(head_input_dim, variable_num, hidden_size=d_model, context_size=(d_model if self.use_context else None), dropout=dropout, use_swiglu=use_swiglu)
                for _ in range(n_selection_heads)
            ])
            # Per-head context projections to de-correlate head inputs
            self.head_context_projections = nn.ModuleList([
                nn.Linear(d_model, d_model)
                for _ in range(n_selection_heads)
            ]) if n_selection_heads > 1 and self.use_context else None
            self.variable_grns = nn.ModuleList([GRN(d_model, d_model, dropout=dropout, use_swiglu=use_swiglu) for _ in range(variable_num)])
            self.feature_gate_grn = None
            self.feature_gate_dropout = None

        if self.use_residual_bypass:
            self.residual_projection = nn.Linear(d_model * variable_num, d_model)
            self.residual_gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.residual_projection = None
            self.register_parameter('residual_gate', None)

    def forward(self, x: Tensor, context: Optional[Tensor] = None, return_weights: bool = False):
        graph_attention = None
        if self.cross_mixing is not None:
            cross_output = self.cross_mixing(x, return_attention=return_weights)
            if return_weights:
                if not isinstance(cross_output, tuple) or len(cross_output) != 2:
                    raise RuntimeError("Dynamic graph mixing must return (features, attention_weights) when return_weights=True.")
                x, graph_attention = cross_output
            else:
                if isinstance(cross_output, tuple):
                    raise RuntimeError("Dynamic graph mixing returned attention weights when return_weights=False.")
                x = cross_output
        # x: [B,T,C,d] or [B,C,d]
        if x.ndim not in (3, 4):
            raise ValueError(f"VSN expects rank-3 or rank-4 input after cross mixing, got shape {tuple(x.shape)}.")
        if x.shape[-2] != len(self.variable_grns):
            raise ValueError(
                f"VSN variable dimension mismatch: expected {len(self.variable_grns)}, got {x.shape[-2]}."
            )
        x_flattened = torch.flatten(x, start_dim=-2)

        # Low-rank projection for high-dimensional inputs
        if self.use_low_rank and self.low_rank_down is not None:
            x_for_heads = self.low_rank_act(self.low_rank_down(x_flattened))
        else:
            x_for_heads = x_flattened

        # Per-feature sigmoid gating path (alternative to softmax selection)
        if self.per_feature_gating and self.feature_gate_grn is not None:
            orig_shape = x.shape  # [B,T,C,d] or [B,C,d]
            C = orig_shape[-2]
            d = orig_shape[-1]
            # Covariate-level sigmoid gates: [..., C]
            covariate_gates = torch.sigmoid(self.feature_gate_grn(x_flattened, context))
            covariate_gates = self.feature_gate_dropout(covariate_gates)
            # Broadcast to [..., C, d] and apply
            gated = x * covariate_gates.unsqueeze(-1)  # [..., C, d]
            selection_result = gated.mean(dim=-2)  # [..., d] — mean keeps scale bounded
            if self.use_residual_bypass:
                residual = self.residual_projection(x_flattened)
                selection_result = selection_result + torch.tanh(self.residual_gate) * residual
            if return_weights:
                return selection_result, {'selection': covariate_gates, 'graph_attention': graph_attention}
            return selection_result

        # x_processed: [B,T,d,C] or [B,d,C]
        x_processed = torch.stack([grn(x[...,i,:]) for i, grn in enumerate(self.variable_grns)], dim=-1)

        K = self.n_selection_heads
        if K == 1:
            # Original single-head path (backward-compatible)
            selection_weights = self.head_grns[0](x_for_heads, context)
            selection_weights = F.softmax(selection_weights, dim=-1)
            selection_result = torch.matmul(x_processed, selection_weights.unsqueeze(-1)).squeeze(-1)
        else:
            # Multi-head: split d dimension into K chunks, each head selects independently
            d_head = x_processed.shape[-2] // K  # d_model // K
            head_results = []
            all_head_weights = []
            for k in range(K):
                head_ctx = self.head_context_projections[k](context) if self.head_context_projections is not None and context is not None else context
                head_weights = self.head_grns[k](x_for_heads, head_ctx)
                head_weights = F.softmax(head_weights, dim=-1)  # [..., C]
                all_head_weights.append(head_weights)
                # Slice processed variables on d dimension for this head's subspace
                chunk = x_processed[..., k * d_head:(k + 1) * d_head, :]  # [..., d_head, C]
                head_result = torch.matmul(chunk, head_weights.unsqueeze(-1)).squeeze(-1)  # [..., d_head]
                head_results.append(head_result)
            selection_result = torch.cat(head_results, dim=-1)  # [..., d_model]
            # Stack per-head weights for interpretation: [..., K, C]
            selection_weights = torch.stack(all_head_weights, dim=-2)

        if self.use_residual_bypass:
            residual = self.residual_projection(x_flattened)
            selection_result = selection_result + torch.tanh(self.residual_gate) * residual
        if return_weights:
            return selection_result, {
                'selection': selection_weights,
                'graph_attention': graph_attention,
            }
        return selection_result


class StaticCovariateEncoder(nn.Module):
    def __init__(self, d_model, static_len, dropout=0.0, use_swiglu=False, cross_variable_mixing=False, n_heads=4, residual_bypass=True, n_selection_heads=1, per_feature_gating=False, low_rank_threshold=64, graph_type='dense', graph_top_k=10, graph_num_layers=2, graph_temporal_evolution=False, graph_edge_features=False):
        super(StaticCovariateEncoder, self).__init__()
        self.canonical_mode = False
        if static_len:
            vsn_kwargs = dict(
                d_model=d_model, variable_num=static_len, dropout=dropout, use_swiglu=use_swiglu,
                cross_variable_mixing=cross_variable_mixing, n_heads=n_heads, residual_bypass=residual_bypass,
                n_selection_heads=n_selection_heads, per_feature_gating=per_feature_gating,
                low_rank_threshold=low_rank_threshold, graph_type=graph_type, graph_top_k=graph_top_k,
                graph_num_layers=graph_num_layers, graph_temporal_evolution=graph_temporal_evolution,
                graph_edge_features=graph_edge_features, use_context=False,
            )
            self.static_vsn_cs = VariableSelectionNetwork(**vsn_kwargs)
            self.static_vsn_cc = VariableSelectionNetwork(**vsn_kwargs)
            self.static_vsn_ch = VariableSelectionNetwork(**vsn_kwargs)
            self.static_vsn_ce = VariableSelectionNetwork(**vsn_kwargs)
        else:
            self.static_vsn_cs = None
            self.static_vsn_cc = None
            self.static_vsn_ch = None
            self.static_vsn_ce = None
        self.static_vsn = None
        self.grns = nn.ModuleList([GRN(d_model, d_model, dropout=dropout, use_swiglu=use_swiglu) for _ in range(4)]) if static_len else None

    def forward(self, static_input, return_weights: bool = False):
        # static_input: [B,C,d]
        if static_input is not None:
            if self.canonical_mode and self.static_vsn is not None:
                if return_weights:
                    feat_shared, w_shared = self.static_vsn(static_input, return_weights=True)
                    shared_contexts = [grn(feat_shared) for grn in self.grns]
                    return shared_contexts, {'c_s': w_shared, 'c_c': w_shared, 'c_h': w_shared, 'c_e': w_shared}
                feat_shared = self.static_vsn(static_input)
                return [grn(feat_shared) for grn in self.grns]
            if return_weights:
                feat_cs, w_cs = self.static_vsn_cs(static_input, return_weights=True)
                feat_cc, w_cc = self.static_vsn_cc(static_input, return_weights=True)
                feat_ch, w_ch = self.static_vsn_ch(static_input, return_weights=True)
                feat_ce, w_ce = self.static_vsn_ce(static_input, return_weights=True)
                feats = [feat_cs, feat_cc, feat_ch, feat_ce]
                return [grn(f) for grn, f in zip(self.grns, feats)], {'c_s': w_cs, 'c_c': w_cc, 'c_h': w_ch, 'c_e': w_ce}
            feat_cs = self.static_vsn_cs(static_input)
            feat_cc = self.static_vsn_cc(static_input)
            feat_ch = self.static_vsn_ch(static_input)
            feat_ce = self.static_vsn_ce(static_input)
            feats = [feat_cs, feat_cc, feat_ch, feat_ce]
            return [grn(f) for grn, f in zip(self.grns, feats)]
        else:
            if return_weights:
                return [None] * 4, None
            return [None] * 4


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, configs):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.n_heads = configs.n_heads
        assert configs.d_model % configs.n_heads == 0
        self.d_head = configs.d_model // configs.n_heads
        self.position_bias_type = getattr(configs, 'tft_attention_position_bias', 'none')
        self.attn_dropout = float(getattr(configs, 'tft_attention_dropout', 0.0))
        self.rope_base = float(getattr(configs, 'tft_rope_base', 10000.0))
        self.alibi_scale = float(getattr(configs, 'tft_alibi_scale', 1.0))
        self.debug_checks = bool(getattr(configs, 'tft_debug_checks', False))
        self.qkv_linears = nn.Linear(configs.d_model, (2 * self.n_heads + 1) * self.d_head, bias=False)
        self.out_projection = nn.Linear(self.d_head, configs.d_model, bias=False)
        self.out_dropout = nn.Dropout(configs.dropout)
        self.scale = self.d_head ** -0.5
        # Pre-build causal mask at max sequence length to avoid re-creation each forward
        max_len = configs.seq_len + configs.pred_len
        self.register_buffer('_causal_mask_buf', build_causal_mask(max_len, torch.device('cpu'), torch.float32), persistent=False)

    def forward(self, x, return_attention: bool = False):
        # Q,K,V are all from x
        B, T, d_model = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_heads * self.d_head, self.n_heads * self.d_head, self.d_head), dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = k.view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = v.view(B, T, self.d_head)

        if self.position_bias_type == 'rope':
            q, k = apply_rotary_embedding(q, k, base=self.rope_base)

        attention_score = torch.matmul(q, k.transpose(-2, -1))  # [B,n,T,T]
        attention_score = attention_score * self.scale
        if self.position_bias_type == 'alibi':
            attention_score = attention_score + build_alibi_bias(
                self.n_heads,
                T,
                T,
                x.device,
                attention_score.dtype,
                scale=self.alibi_scale,
            )
        if self.debug_checks and not torch.isfinite(attention_score).all():
            raise ValueError("Unmasked attention scores contain NaN/Inf values.")
        clamp_limit = 1e4
        if self.debug_checks and (attention_score.abs() > clamp_limit).any():
            warnings.warn("Attention scores exceeded the stability clamp threshold; values were clipped.")
            attention_score = attention_score.clamp(min=-clamp_limit, max=clamp_limit)
        if T <= self._causal_mask_buf.shape[0]:
            causal_mask = self._causal_mask_buf[:T, :T].to(device=attention_score.device, dtype=attention_score.dtype)
        else:
            causal_mask = build_causal_mask(T, attention_score.device, attention_score.dtype)
        attention_score = attention_score + causal_mask
        attention_prob = F.softmax(attention_score, dim=3)  # [B,n,T,T]
        if self.debug_checks and not torch.isfinite(attention_prob).all():
            raise ValueError("Attention probabilities contain NaN/Inf values.")
        attention_prob_used = F.dropout(attention_prob, p=self.attn_dropout, training=self.training)

        attention_out = torch.matmul(attention_prob_used, v.unsqueeze(1))  # [B,n,T,d]
        attention_out = torch.mean(attention_out, dim=1)  # [B,T,d]
        out = self.out_projection(attention_out)
        out = self.out_dropout(out)  # [B,T,d]
        if return_attention:
            return out, attention_prob
        return out


class TemporalFusionDecoderLayer(nn.Module):
    def __init__(self, configs):
        super(TemporalFusionDecoderLayer, self).__init__()
        self.pred_len = configs.pred_len
        self.use_swiglu = getattr(configs, 'tft_use_swiglu', False)
        self.full_attention = getattr(configs, 'tft_full_attention', True)
        self.dual_attention_fusion = getattr(configs, 'tft_dual_attention_fusion', False)
        self.use_explicit_cross_attention = getattr(configs, 'tft_use_explicit_cross_attention', False)
        self.cross_attention_type = getattr(configs, 'tft_cross_attention_type', 'full')
        self.position_bias_type = getattr(configs, 'tft_attention_position_bias', 'none')
        self.attention_backend = getattr(configs, 'tft_attention_backend', 'exact')
        self.debug_checks = bool(getattr(configs, 'tft_debug_checks', False))
        self.attn_dropout = float(getattr(configs, 'tft_attention_dropout', 0.0))
        self.rope_base = float(getattr(configs, 'tft_rope_base', 10000.0))
        self.alibi_scale = float(getattr(configs, 'tft_alibi_scale', 1.0))
        self.use_lag_attention = getattr(configs, 'tft_use_lag_attention', False)
        _lag_raw = getattr(configs, 'tft_lag_scales', [1, 2, 4, 8])
        if isinstance(_lag_raw, str):
            self.lag_scales = [int(x.strip()) for x in _lag_raw.split(',') if x.strip()]
        elif isinstance(_lag_raw, (list, tuple)):
            self.lag_scales = [int(x) for x in _lag_raw]
        else:
            self.lag_scales = [1, 2, 4, 8]
        self.temporal_backbone_type = getattr(configs, 'tft_temporal_backbone', 'hybrid_tcn_lstm')
        self.temporal_backbone_layers = int(getattr(configs, 'tft_temporal_backbone_layers', 3))
        self.temporal_kernel_size = int(getattr(configs, 'tft_temporal_kernel_size', 3))
        self.temporal_hidden_size = getattr(configs, 'tft_temporal_hidden_size', 0) or configs.d_model
        self.use_higher_order = getattr(configs, 'tft_use_higher_order', False)
        self.interaction_order = int(getattr(configs, 'tft_interaction_order', 2))
        self.interaction_rank = getattr(configs, 'tft_interaction_rank', 0) or None
        self.use_regime_moe = getattr(configs, 'tft_use_regime_moe', False)
        self.num_regimes = int(getattr(configs, 'tft_num_regimes', 4))
        self.num_moe_experts = int(getattr(configs, 'tft_num_moe_experts', 4))
        self.moe_top_k = int(getattr(configs, 'tft_moe_top_k', 2))
        self.moe_hidden_size = getattr(configs, 'tft_moe_hidden_size', 0) or configs.d_model
        self.moe_noise_epsilon = float(getattr(configs, 'tft_moe_noise_epsilon', 1e-2))
        self.moe_capacity_factor = float(getattr(configs, 'tft_moe_capacity_factor', 1.25))
        self.use_fft_branch = getattr(configs, 'tft_use_fft_branch', False)
        self.fft_modes = int(getattr(configs, 'tft_fft_modes', 32))
        self.fft_mode_select = getattr(configs, 'tft_fft_mode_select', 'low')
        self.use_temporal_compression = getattr(configs, 'tft_use_temporal_compression', False)
        self.tc_stride = int(getattr(configs, 'tft_tc_stride', 2))
        self.tc_threshold = int(getattr(configs, 'tft_tc_threshold', 256))

        if self.temporal_backbone_type == 'lstm':
            self.history_encoder = nn.LSTM(configs.d_model, configs.d_model, batch_first=True)
            self.future_encoder = nn.LSTM(configs.d_model, configs.d_model, batch_first=True)
            self.temporal_backbone = None
        elif self.temporal_backbone_type == 'gated_tcn':
            self.history_encoder = None
            self.future_encoder = None
            self.temporal_backbone = GatedDilatedTemporalBackbone(
                configs.d_model,
                num_layers=self.temporal_backbone_layers,
                kernel_size=self.temporal_kernel_size,
                hidden_size=self.temporal_hidden_size,
                dropout=configs.dropout,
                debug_checks=self.debug_checks,
            )
        elif self.temporal_backbone_type == 'hybrid_tcn_lstm':
            self.history_encoder = None
            self.future_encoder = None
            self.temporal_backbone = HybridTemporalBackbone(
                configs.d_model,
                num_layers=self.temporal_backbone_layers,
                kernel_size=self.temporal_kernel_size,
                hidden_size=self.temporal_hidden_size,
                dropout=configs.dropout,
                debug_checks=self.debug_checks,
            )
        else:
            raise ValueError("tft_temporal_backbone must be one of: lstm, gated_tcn, hybrid_tcn_lstm.")
        if self.use_fft_branch:
            self.fft_branch = SpectralBranch(
                configs.d_model,
                modes=self.fft_modes,
                mode_select=self.fft_mode_select,
                dropout=configs.dropout,
            )
            self.fft_fusion_gate = nn.Linear(configs.d_model * 2, configs.d_model)
        else:
            self.fft_branch = None
            self.fft_fusion_gate = None
        # TEMPORAL COMPRESSION (optional, learned stride for long sequences)
        if self.use_temporal_compression:
            self.temporal_compression = TemporalCompression(
                configs.d_model,
                stride=self.tc_stride,
                threshold=self.tc_threshold,
                dropout=configs.dropout,
            )
        else:
            self.temporal_compression = None
        self.gate_after_lstm = GateAddNorm(configs.d_model, configs.d_model, use_swiglu=self.use_swiglu)
        self.enrichment_grn = GRN(configs.d_model, configs.d_model, context_size=configs.d_model, dropout=configs.dropout, use_swiglu=self.use_swiglu)
        if self.use_explicit_cross_attention:
            if self.cross_attention_type == 'interpretable':
                self.cross_attention = InterpretableCrossAttention(
                    configs.d_model,
                    configs.n_heads,
                    dropout=configs.dropout,
                    position_bias_type=self.position_bias_type,
                    rope_base=self.rope_base,
                    alibi_scale=self.alibi_scale,
                    debug_checks=self.debug_checks,
                    attn_dropout=self.attn_dropout,
                )
            else:
                self.cross_attention = PositionalMultiHeadAttention(
                    configs.d_model,
                    configs.n_heads,
                    dropout=configs.dropout,
                    position_bias_type=self.position_bias_type,
                    rope_base=self.rope_base,
                    alibi_scale=self.alibi_scale,
                    attention_backend=self.attention_backend,
                    debug_checks=self.debug_checks,
                    attn_dropout=self.attn_dropout,
                )
        else:
            self.cross_attention = None
        self.gate_after_cross_attention = GateAddNorm(configs.d_model, configs.d_model, use_swiglu=self.use_swiglu) if self.use_explicit_cross_attention else None
        self.lag_attention_module = MultiScaleLagAttention(
            configs.d_model,
            configs.n_heads,
            self.lag_scales,
            dropout=configs.dropout,
            position_bias_type=self.position_bias_type,
            rope_base=self.rope_base,
            alibi_scale=self.alibi_scale,
            attention_backend=self.attention_backend,
            max_seq_len=configs.seq_len + configs.pred_len,
            debug_checks=self.debug_checks,
            attn_dropout=self.attn_dropout,
        ) if self.use_lag_attention else None
        self.higher_order_block = HigherOrderInteractionBlock(
            configs.d_model,
            interaction_order=self.interaction_order,
            interaction_rank=self.interaction_rank,
            dropout=configs.dropout,
            debug_checks=self.debug_checks,
        ) if self.use_higher_order else None
        self.regime_moe = RegimeAwareSparseMoE(
            configs.d_model,
            num_experts=self.num_moe_experts,
            top_k=self.moe_top_k,
            num_regimes=self.num_regimes,
            hidden_size=self.moe_hidden_size,
            dropout=configs.dropout,
            noise_epsilon=self.moe_noise_epsilon,
        ) if self.use_regime_moe else None
        if self.regime_moe is not None:
            self.regime_moe.capacity_factor = self.moe_capacity_factor
            self.regime_moe.debug_checks = self.debug_checks
        if self.dual_attention_fusion:
            self.full_attention_module = PositionalMultiHeadAttention(
                configs.d_model,
                configs.n_heads,
                dropout=configs.dropout,
                position_bias_type=self.position_bias_type,
                rope_base=self.rope_base,
                alibi_scale=self.alibi_scale,
                attention_backend=self.attention_backend,
                debug_checks=self.debug_checks,
                attn_dropout=self.attn_dropout,
            )
            self.interpretable_attention_module = InterpretableMultiHeadAttention(configs)
        elif self.full_attention:
            self.attention = PositionalMultiHeadAttention(
                configs.d_model,
                configs.n_heads,
                dropout=configs.dropout,
                position_bias_type=self.position_bias_type,
                rope_base=self.rope_base,
                alibi_scale=self.alibi_scale,
                attention_backend=self.attention_backend,
                debug_checks=self.debug_checks,
                attn_dropout=self.attn_dropout,
            )
        else:
            self.attention = InterpretableMultiHeadAttention(configs)
        branch_count = 2 if self.dual_attention_fusion else 1
        if self.use_lag_attention:
            branch_count += 1
        if branch_count > 1:
            # Bias toward the proven branch so supplementary branches must earn
            # weight during training.  For dual-attention the interpretable head
            # (index 1) is the baseline; otherwise the main head is index 0.
            init_logits = torch.full((branch_count,), -1.0)
            main_idx = 1 if self.dual_attention_fusion else 0
            init_logits[main_idx] = 0.0
            self.attention_fusion_logits = nn.Parameter(init_logits)
        else:
            self.attention_fusion_logits = None
        self.gate_after_attention = GateAddNorm(configs.d_model, configs.d_model, use_swiglu=self.use_swiglu)
        self.position_wise_grn = GRN(configs.d_model, configs.d_model, dropout=configs.dropout, use_swiglu=self.use_swiglu) if not self.use_regime_moe else None
        self.gate_final = GateAddNorm(configs.d_model, configs.d_model, use_swiglu=self.use_swiglu)
        self.last_moe_aux_loss = None
        self.last_moe_importance_sum = None
        self.last_moe_load_sum = None
        self.last_moe_token_count = None
        # Covariate-aware enrichment: cross-attention to pre-VSN covariate embeddings
        self.use_covariate_reattention = getattr(configs, 'tft_covariate_reattention', False)
        if self.use_covariate_reattention:
            self.covariate_cross_attention = PositionalMultiHeadAttention(
                configs.d_model, configs.n_heads,
                position_bias_type='none',
                attention_backend='exact',
                debug_checks=self.debug_checks,
                attn_dropout=self.attn_dropout,
            )
            self.gate_after_reattention = GateAddNorm(configs.d_model, configs.d_model, use_swiglu=self.use_swiglu)
        # Pre-build causal mask at max sequence length to avoid re-creation each forward
        max_len = configs.seq_len + configs.pred_len
        self.register_buffer('_causal_mask_buf', build_causal_mask(max_len, torch.device('cpu'), torch.float32), persistent=False)

    def forward(self, history_input, future_input, c_c, c_h, c_e, return_attention: bool = False, pre_vsn_embs=None):
        self.last_moe_aux_loss = None
        self.last_moe_importance_sum = None
        self.last_moe_load_sum = None
        self.last_moe_token_count = None
        output_payload = {} if return_attention else None
        temporal_input = torch.cat([history_input, future_input], dim=1)
        if self.temporal_backbone_type == 'lstm':
            # PyTorch recurrent modules expect state order (h_0, c_0).
            initial_hidden = c_h.unsqueeze(0) if c_h is not None else None
            initial_cell = c_c.unsqueeze(0) if c_c is not None else None
            initial_state = (initial_hidden, initial_cell) if initial_hidden is not None and initial_cell is not None else None
            historical_features, state = self.history_encoder(history_input, initial_state)
            future_features, _ = self.future_encoder(future_input, state)
            temporal_features = torch.cat([historical_features, future_features], dim=1)
        elif self.temporal_backbone_type == 'gated_tcn':
            temporal_features = self.temporal_backbone(temporal_input)
        else:
            # HybridTemporalBackbone follows the same (h_0, c_0) contract.
            initial_hidden = c_h.unsqueeze(0) if c_h is not None else None
            initial_cell = c_c.unsqueeze(0) if c_c is not None else None
            initial_state = (initial_hidden, initial_cell) if initial_hidden is not None and initial_cell is not None else None
            temporal_features, _ = self.temporal_backbone(temporal_input, state=initial_state)
        # FFT spectral branch: parallel to temporal backbone, fused via learned gate
        if self.use_fft_branch:
            fft_features = self.fft_branch(temporal_input)
            fft_gate = torch.sigmoid(self.fft_fusion_gate(
                torch.cat([temporal_features, fft_features], dim=-1)
            ))
            temporal_features = fft_gate * temporal_features + (1.0 - fft_gate) * fft_features
            if return_attention:
                output_payload['fft_gate_mean'] = fft_gate.mean().item()
                learned_mask_summary = self.fft_branch.summarize_learned_mask(
                    n_freqs=(temporal_input.shape[1] // 2) + 1
                )
                if learned_mask_summary is not None:
                    output_payload.update(learned_mask_summary)
        temporal_features = self.gate_after_lstm(temporal_features, temporal_input)

        # Covariate-aware reattention: let temporal features attend to pre-VSN covariate embeddings
        if self.use_covariate_reattention and pre_vsn_embs is not None:
            B_cov, T_cov, C_cov, d_cov = pre_vsn_embs.shape
            kv_tokens = pre_vsn_embs.reshape(B_cov, T_cov * C_cov, d_cov)
            reattended = self.covariate_cross_attention(
                temporal_features, kv_tokens, kv_tokens,
            )
            temporal_features = self.gate_after_reattention(reattended, temporal_features)

        # TEMPORAL COMPRESSION: compress history portion for long sequences
        history_len = history_input.shape[1]
        _tc_active = (self.temporal_compression is not None
                      and self.temporal_compression.should_compress(history_len))
        if _tc_active:
            hist_feats = temporal_features[:, :history_len, :]
            fut_feats = temporal_features[:, history_len:, :]
            hist_compressed, _tc_orig_len = self.temporal_compression.compress(hist_feats)
            # Recombine with full-resolution future
            temporal_features_for_attn = torch.cat([hist_compressed, fut_feats], dim=1)
            compressed_history_len = hist_compressed.shape[1]
            history_positions = torch.arange(
                0,
                history_len,
                step=self.temporal_compression.stride,
                device=temporal_features.device,
            )[:compressed_history_len]
            future_positions = torch.arange(history_len, history_len + fut_feats.shape[1], device=temporal_features.device)
            temporal_positions = torch.cat([history_positions, future_positions], dim=0)
            if return_attention:
                output_payload['tc_active'] = True
                output_payload['tc_compressed_history_len'] = compressed_history_len
        else:
            temporal_features_for_attn = temporal_features
            compressed_history_len = history_len
            temporal_positions = torch.arange(temporal_features_for_attn.shape[1], device=temporal_features_for_attn.device)
            if return_attention and self.temporal_compression is not None:
                output_payload['tc_active'] = False

        enriched_features = self.enrichment_grn(temporal_features_for_attn, c_e)
        if self.use_explicit_cross_attention:
            enriched_history = enriched_features[:, :compressed_history_len, :]
            enriched_future = enriched_features[:, compressed_history_len:, :]
            history_positions = temporal_positions[:compressed_history_len]
            future_positions = temporal_positions[compressed_history_len:]
            if self.cross_attention_type == 'interpretable':
                if return_attention:
                    cross_out, cross_attention_prob = self.cross_attention(
                        enriched_future,
                        enriched_history,
                        return_attention=True,
                        query_positions=future_positions,
                        key_positions=history_positions,
                    )
                else:
                    cross_out = self.cross_attention(
                        enriched_future,
                        enriched_history,
                        query_positions=future_positions,
                        key_positions=history_positions,
                    )
                    cross_attention_prob = None
            else:
                if return_attention:
                    cross_out, cross_attention_prob = self.cross_attention(
                        enriched_future,
                        enriched_history,
                        enriched_history,
                        return_attention=True,
                        query_positions=future_positions,
                        key_positions=history_positions,
                    )
                else:
                    cross_out = self.cross_attention(
                        enriched_future,
                        enriched_history,
                        enriched_history,
                        query_positions=future_positions,
                        key_positions=history_positions,
                    )
                    cross_attention_prob = None
            enriched_future = self.gate_after_cross_attention(cross_out, enriched_future)
            enriched_features = torch.cat([enriched_history, enriched_future], dim=1)
            if return_attention:
                output_payload['cross_attention'] = cross_attention_prob
                output_payload['cross_attention_type'] = self.cross_attention_type
        seq_len = enriched_features.shape[1]
        if seq_len <= self._causal_mask_buf.shape[0]:
            attn_mask = self._causal_mask_buf[:seq_len, :seq_len].to(device=enriched_features.device, dtype=enriched_features.dtype)
        else:
            attn_mask = build_causal_mask(seq_len, enriched_features.device, enriched_features.dtype)
        attention_branches = []

        if self.dual_attention_fusion:
            if return_attention:
                full_out, full_attention_prob = self.full_attention_module(
                    enriched_features,
                    enriched_features,
                    enriched_features,
                    return_attention=True,
                    attn_mask=attn_mask,
                )
            else:
                full_out = self.full_attention_module(
                    enriched_features,
                    enriched_features,
                    enriched_features,
                    attn_mask=attn_mask,
                )
                full_attention_prob = None

            if return_attention:
                interpretable_out, interpretable_attention_prob = self.interpretable_attention_module(
                    enriched_features, return_attention=True
                )
            else:
                interpretable_out = self.interpretable_attention_module(enriched_features)
                interpretable_attention_prob = None

            attention_branches.extend([full_out, interpretable_out])
            if return_attention:
                output_payload['interpretable'] = interpretable_attention_prob
                output_payload['full'] = full_attention_prob
        elif self.full_attention:
            if return_attention:
                attention_out, attention_prob = self.attention(
                    enriched_features,
                    enriched_features,
                    enriched_features,
                    return_attention=True,
                    attn_mask=attn_mask,
                )
            else:
                attention_out = self.attention(
                    enriched_features,
                    enriched_features,
                    enriched_features,
                    attn_mask=attn_mask,
                )
                attention_prob = None
            attention_branches.append(attention_out)
            if return_attention:
                output_payload['full'] = attention_prob
        else:
            if return_attention:
                attention_out, attention_prob = self.attention(enriched_features, return_attention=True)
            else:
                attention_out = self.attention(enriched_features)
                attention_prob = None
            attention_branches.append(attention_out)
            if return_attention:
                output_payload['interpretable'] = attention_prob

        if self.use_lag_attention:
            if return_attention:
                lag_out, lag_payload = self.lag_attention_module(
                    enriched_features,
                    return_attention=True,
                    positions=temporal_positions,
                )
                output_payload.update(lag_payload)
            else:
                lag_out = self.lag_attention_module(enriched_features, positions=temporal_positions)
            attention_branches.append(lag_out)

        if len(attention_branches) == 1:
            attention_out = attention_branches[0]
            branch_weights = None
        else:
            branch_weights = torch.softmax(self.attention_fusion_logits, dim=0)
            attention_out = sum(weight * branch for weight, branch in zip(branch_weights, attention_branches))

        if return_attention:
            output_payload['temporal_backbone_type'] = self.temporal_backbone_type
            output_payload['position_bias_type'] = self.position_bias_type
            output_payload['temporal_positions'] = temporal_positions.detach()
            output_payload['attention_backend_config'] = self.attention_backend
            if self.full_attention:
                attention_backend_used = self.full_attention_module.last_attention_backend if self.dual_attention_fusion else self.attention.last_attention_backend
                output_payload['attention_backend_used'] = attention_backend_used
            if self.use_explicit_cross_attention and hasattr(self.cross_attention, 'last_attention_backend'):
                output_payload['cross_attention_backend_used'] = self.cross_attention.last_attention_backend
            if branch_weights is not None:
                output_payload['attention_branch_weights'] = branch_weights.detach()
                if self.dual_attention_fusion and not self.use_lag_attention and branch_weights.numel() == 2:
                    output_payload['fusion_alpha'] = branch_weights[0].detach()
                else:
                    output_payload['fusion_alpha'] = branch_weights.detach()
            else:
                output_payload['fusion_alpha'] = None

        attention_out = self.gate_after_attention(attention_out, enriched_features)
        # TEMPORAL DECOMPRESSION: restore history to original length
        if _tc_active:
            compressed_hist_out = attention_out[:, :compressed_history_len, :]
            fut_out = attention_out[:, compressed_history_len:, :]
            hist_restored = self.temporal_compression.decompress_to(compressed_hist_out, _tc_orig_len)
            attention_out = torch.cat([hist_restored, fut_out], dim=1)
        if self.use_higher_order:
            if return_attention:
                attention_out, interaction_payload = self.higher_order_block(attention_out, return_payload=True)
                output_payload.update(interaction_payload)
            else:
                attention_out = self.higher_order_block(attention_out)
        if self.use_regime_moe:
            attention_out, moe_aux_loss, moe_payload = self.regime_moe(attention_out, context=c_e, return_payload=True)
            self.last_moe_importance_sum = moe_payload.get('expert_importance_sum')
            self.last_moe_load_sum = moe_payload.get('expert_load_sum')
            self.last_moe_token_count = moe_payload.get('expert_token_count')
            if return_attention:
                output_payload.update(moe_payload)
                output_payload['moe_aux_loss'] = moe_aux_loss.detach()
            self.last_moe_aux_loss = moe_aux_loss
            out = attention_out
        else:
            out = self.position_wise_grn(attention_out)
        out = self.gate_final(out, temporal_features)
        if return_attention:
            return out, output_payload
        return out


class TemporalFusionDecoder(nn.Module):
    def __init__(self, configs):
        super(TemporalFusionDecoder, self).__init__()
        self.e_layers = getattr(configs, 'e_layers', 1)
        self.pred_len = configs.pred_len
        configured_output_mode = getattr(configs, 'tft_output_mode', None)
        if configured_output_mode is None:
            configured_output_mode = 'joint' if getattr(configs, 'tft_use_quantile_head', False) else 'point'
        self.output_mode = str(configured_output_mode).lower()
        self.stack_payload_layers = getattr(configs, 'tft_payload_stack_layers', True)
        self.stochastic_depth_rate = float(getattr(configs, 'tft_stochastic_depth_rate', 0.0))
        self.gradient_checkpointing = getattr(configs, 'tft_gradient_checkpointing', False)
        self.layers = nn.ModuleList([TemporalFusionDecoderLayer(configs) for _ in range(self.e_layers)])
        self.has_point_head = self.output_mode in {'point', 'joint'}
        self.use_per_target_heads = getattr(configs, 'tft_per_target_heads', False)
        if self.has_point_head and self.use_per_target_heads and configs.c_out > 1:
            head_hidden = max(configs.d_model // 2, 1)
            self.target_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(configs.d_model, head_hidden),
                    nn.GELU(),
                    nn.Linear(head_hidden, 1),
                )
                for _ in range(configs.c_out)
            ])
            self.out_projection = None
        elif self.has_point_head:
            self.out_projection = nn.Linear(configs.d_model, configs.c_out)
            self.target_heads = None
        else:
            self.out_projection = None
            self.target_heads = None
        self.last_moe_aux_loss = None
        self.last_moe_importance_sum = None
        self.last_moe_load_sum = None
        self.last_moe_token_count = None

    def _aggregate_attention_payloads(self, payloads):
        if not payloads:
            return None
        final_payload = payloads[-1]
        if not isinstance(final_payload, dict):
            return final_payload

        merged_payload = dict(final_payload)
        merged_payload['decoder_num_layers'] = len(payloads)
        if not self.stack_payload_layers:
            return merged_payload

        layerwise = {}
        keys = set().union(*(payload.keys() for payload in payloads if isinstance(payload, dict)))
        for key in keys:
            values = [payload.get(key) if isinstance(payload, dict) else None for payload in payloads]
            if all(value is None for value in values):
                layerwise[key] = None
            elif all(torch.is_tensor(value) for value in values):
                try:
                    layerwise[key] = torch.stack(values, dim=0)
                except RuntimeError:
                    layerwise[key] = values
            else:
                layerwise[key] = values
        merged_payload['decoder_layer_payloads'] = layerwise
        return merged_payload

    def forward(self, history_input, future_input, c_c, c_h, c_e, return_attention: bool = False, return_decoder_hidden: bool = False, pre_vsn_embs=None):
        attention_payloads = []
        moe_aux_losses = []
        moe_importance_sums = []
        moe_load_sums = []
        moe_token_counts = []
        curr_history = history_input
        curr_future = future_input
        num_layers = len(self.layers)
        out = torch.cat([curr_history, curr_future], dim=1)

        def _layer_forward(layer_module, h, f, cc, ch, ce):
            return layer_module(h, f, cc, ch, ce, pre_vsn_embs=pre_vsn_embs)

        for layer_idx, layer in enumerate(self.layers):
            # Stochastic depth: skip layers with linearly increasing probability
            if self.training and self.stochastic_depth_rate > 0.0 and num_layers > 1 and layer_idx > 0:
                drop_prob = layer_idx / (num_layers - 1) * self.stochastic_depth_rate
                if torch.rand(1).item() < drop_prob:
                    # Identity pass-through: keep curr_history/curr_future unchanged
                    out = torch.cat([curr_history, curr_future], dim=1)
                    # Must update splits so the next layer receives the correct tensors
                    curr_history = out[:, :history_input.shape[1], :]
                    curr_future = out[:, history_input.shape[1]:, :]
                    if return_attention:
                        attention_payloads.append({})
                    continue

            # Gradient checkpointing: trade memory for compute during training
            if self.gradient_checkpointing and self.training and not return_attention:
                out = torch.utils.checkpoint.checkpoint(
                    _layer_forward, layer, curr_history, curr_future, c_c, c_h, c_e,
                    use_reentrant=False,
                )
            elif return_attention:
                out, attention_prob = layer(curr_history, curr_future, c_c, c_h, c_e, return_attention=True, pre_vsn_embs=pre_vsn_embs)
                attention_payloads.append(attention_prob)
            else:
                out = layer(curr_history, curr_future, c_c, c_h, c_e, pre_vsn_embs=pre_vsn_embs)

            if layer.last_moe_aux_loss is not None:
                moe_aux_losses.append(layer.last_moe_aux_loss)
            if layer.last_moe_importance_sum is not None:
                moe_importance_sums.append(layer.last_moe_importance_sum)
            if layer.last_moe_load_sum is not None:
                moe_load_sums.append(layer.last_moe_load_sum)
            if layer.last_moe_token_count is not None:
                moe_token_counts.append(layer.last_moe_token_count)
            curr_history = out[:, :history_input.shape[1], :]
            curr_future = out[:, history_input.shape[1]:, :]

        self.last_moe_aux_loss = torch.stack(moe_aux_losses).mean() if moe_aux_losses else None
        self.last_moe_importance_sum = torch.stack(moe_importance_sums, dim=0).sum(dim=0) if moe_importance_sums else None
        self.last_moe_load_sum = torch.stack(moe_load_sums, dim=0).sum(dim=0) if moe_load_sums else None
        self.last_moe_token_count = torch.stack(moe_token_counts, dim=0).sum(dim=0) if moe_token_counts else None
        decoder_hidden = out[:, -self.pred_len:, :]
        if self.target_heads is not None:
            projected = torch.cat([head(decoder_hidden) for head in self.target_heads], dim=-1)
        elif self.out_projection is not None:
            projected = self.out_projection(decoder_hidden)
        else:
            projected = None

        if return_attention:
            payload = self._aggregate_attention_payloads(attention_payloads)
            if return_decoder_hidden:
                return projected, payload, decoder_hidden
            return projected, payload
        if return_decoder_hidden:
            return projected, decoder_hidden
        return projected


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = apply_tft_profile(configs)
        configs = self.configs
        self.task_name = configs.task_name
        if self.task_name == 'short_term_forecast':
            raise NotImplementedError(
                "TemporalFusionTransformer currently supports long_term_forecast only. "
                "short_term_forecast/M4 mark semantics are not implemented for the native TFT path."
            )
        if self.task_name != 'long_term_forecast':
            raise NotImplementedError(
                f"TemporalFusionTransformer supports long_term_forecast only, got task_name={self.task_name!r}."
            )
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        configs._tft_resolved_schema = resolve_tft_schema(configs)
        self.resolved_tft_schema = configs._tft_resolved_schema

        # Number of variables
        typepos = get_typepos(configs)
        self.static_len = len(typepos.static)
        self.observed_len = len(typepos.observed)
        self.target_pos = get_target_pos(configs)
        self.register_buffer('target_pos_buf', torch.tensor(self.target_pos, dtype=torch.long), persistent=False)
        self.allow_custom_known = getattr(configs, 'tft_allow_custom_known', False)
        self.use_revin = getattr(configs, 'tft_use_revin', False)
        self.revin_affine = getattr(configs, 'tft_revin_affine', True)
        requested_quantile_head = getattr(configs, 'tft_use_quantile_head', False)
        configured_output_mode = getattr(configs, 'tft_output_mode', None)
        if configured_output_mode is None:
            configured_output_mode = 'joint' if requested_quantile_head else 'point'
        self.output_mode = str(configured_output_mode).lower()
        if self.output_mode not in {'point', 'quantile', 'joint'}:
            raise ValueError("tft_output_mode must be one of: point, quantile, joint.")
        self.use_quantile_head = self.output_mode in {'quantile', 'joint'}
        self.tft_profile = getattr(configs, 'tft_profile', 'extended_safe')
        self.is_canonical_profile = self.tft_profile == 'canonical'
        self.debug_checks = bool(getattr(configs, 'tft_debug_checks', False))
        self.position_bias_type = getattr(configs, 'tft_attention_position_bias', 'none')
        self.temporal_backbone_type = getattr(configs, 'tft_temporal_backbone', 'hybrid_tcn_lstm')
        self.attention_backend = getattr(configs, 'tft_attention_backend', 'exact')
        quantiles = getattr(configs, 'tft_output_quantiles', [0.1, 0.5, 0.9])
        self.quantiles = None
        self.quantile_median_index = None
        if self.allow_custom_known:
            self.known_len = len(self.resolved_tft_schema.known_feature_names)
        else:
            self.known_len = len(self.resolved_tft_schema.known_feature_names)

        self.embedding = TFTEmbedding(configs)
        self.revin = Normalize(configs.enc_in, affine=self.revin_affine) if self.use_revin else None
        
        self.use_swiglu = getattr(configs, 'tft_use_swiglu', False)
        self.cross_mix = getattr(configs, 'tft_cross_variable_mixing', False)
        self.vsn_residual_bypass = getattr(configs, 'tft_vsn_residual_bypass', True)
        self.n_heads = getattr(configs, 'n_heads', 4)
        self.n_selection_heads = int(getattr(configs, 'tft_vsn_n_selection_heads', 1))
        self.per_feature_gating = getattr(configs, 'tft_vsn_per_feature_gating', None)
        if self.per_feature_gating is None:
            self.per_feature_gating = ((self.observed_len + self.known_len) >= 10)
        self.vsn_low_rank_threshold = int(getattr(configs, 'tft_vsn_low_rank_threshold', 64))
        self.use_covariate_reattention = getattr(configs, 'tft_covariate_reattention', False)
        self.graph_type = getattr(configs, 'tft_graph_type', 'dense')
        self.graph_top_k = int(getattr(configs, 'tft_graph_top_k', 10))
        self.graph_num_layers = int(getattr(configs, 'tft_graph_num_layers', 2))
        self.graph_temporal_evolution = getattr(configs, 'tft_graph_temporal_evolution', False)
        self.graph_edge_features = getattr(configs, 'tft_graph_edge_features', False)
        self.last_moe_aux_loss = None
        self.last_moe_importance_sum = None
        self.last_moe_load_sum = None
        self.last_moe_token_count = None

        self.static_encoder = StaticCovariateEncoder(
            configs.d_model, self.static_len, dropout=configs.dropout, 
            use_swiglu=self.use_swiglu, cross_variable_mixing=self.cross_mix, n_heads=self.n_heads,
            residual_bypass=self.vsn_residual_bypass, n_selection_heads=self.n_selection_heads,
            per_feature_gating=self.per_feature_gating,
            low_rank_threshold=self.vsn_low_rank_threshold,
            graph_type=self.graph_type, graph_top_k=self.graph_top_k,
            graph_num_layers=self.graph_num_layers, graph_temporal_evolution=self.graph_temporal_evolution,
            graph_edge_features=self.graph_edge_features
        )
        if self.is_canonical_profile and self.static_len:
            self.static_encoder.canonical_mode = True
            self.static_encoder.static_vsn = VariableSelectionNetwork(
                configs.d_model, self.static_len, dropout=configs.dropout,
                use_swiglu=self.use_swiglu, cross_variable_mixing=self.cross_mix, n_heads=self.n_heads,
                residual_bypass=self.vsn_residual_bypass, n_selection_heads=self.n_selection_heads,
                per_feature_gating=self.per_feature_gating,
                low_rank_threshold=self.vsn_low_rank_threshold,
                graph_type=self.graph_type, graph_top_k=self.graph_top_k,
                graph_num_layers=self.graph_num_layers, graph_temporal_evolution=self.graph_temporal_evolution,
                graph_edge_features=self.graph_edge_features, use_context=False,
            )
            self.static_encoder.static_vsn_cs = None
            self.static_encoder.static_vsn_cc = None
            self.static_encoder.static_vsn_ch = None
            self.static_encoder.static_vsn_ce = None
        self.history_vsn = VariableSelectionNetwork(
            configs.d_model, self.observed_len + self.known_len, dropout=configs.dropout,
            use_swiglu=self.use_swiglu, cross_variable_mixing=self.cross_mix, n_heads=self.n_heads,
            residual_bypass=self.vsn_residual_bypass, n_selection_heads=self.n_selection_heads,
            per_feature_gating=self.per_feature_gating,
            low_rank_threshold=self.vsn_low_rank_threshold,
            graph_type=self.graph_type, graph_top_k=self.graph_top_k,
            graph_num_layers=self.graph_num_layers, graph_temporal_evolution=self.graph_temporal_evolution,
            graph_edge_features=self.graph_edge_features
        )
        self.future_vsn = VariableSelectionNetwork(
            configs.d_model, self.known_len, dropout=configs.dropout,
            use_swiglu=self.use_swiglu, cross_variable_mixing=self.cross_mix, n_heads=self.n_heads,
            residual_bypass=self.vsn_residual_bypass, n_selection_heads=self.n_selection_heads,
            per_feature_gating=self.per_feature_gating,
            low_rank_threshold=self.vsn_low_rank_threshold,
            graph_type=self.graph_type, graph_top_k=self.graph_top_k,
            graph_num_layers=self.graph_num_layers, graph_temporal_evolution=self.graph_temporal_evolution,
            graph_edge_features=self.graph_edge_features
        )
        self.temporal_fusion_decoder = TemporalFusionDecoder(configs)
        if self.use_quantile_head:
            self.quantiles = list(canonicalize_quantiles(quantiles))
            if self.output_mode == 'quantile':
                if 0.5 not in self.quantiles:
                    raise ValueError("tft_output_mode=quantile requires quantile level 0.5 to derive the point forecast.")
                self.quantile_median_index = self.quantiles.index(0.5)
            q_out_dim = configs.c_out * len(self.quantiles)
            use_mlp_quantile = getattr(configs, 'tft_mlp_quantile_projection', False)
            if use_mlp_quantile:
                q_ff_size = getattr(configs, 'tft_quantile_projection_ff_size', None)
                q_ff_size = configs.d_model if q_ff_size is None or q_ff_size <= 0 else int(q_ff_size)
                self.quantile_projection = nn.Sequential(
                    nn.Linear(configs.d_model, q_ff_size),
                    nn.GELU(),
                    nn.Dropout(configs.dropout),
                    nn.Linear(q_ff_size, q_out_dim),
                )
            else:
                self.quantile_projection = nn.Linear(configs.d_model, q_out_dim)
        else:
            self.quantile_projection = None
        self.last_quantile_predictions = None

    def _validate_inputs(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if x_enc.ndim != 3 or x_mark_enc.ndim != 3 or x_dec.ndim != 3 or x_mark_dec.ndim != 3:
            raise ValueError("All TFT inputs must be rank-3 tensors [B,T,D].")
        if x_enc.shape[0] != x_mark_enc.shape[0] or x_enc.shape[0] != x_dec.shape[0] or x_enc.shape[0] != x_mark_dec.shape[0]:
            raise ValueError("Batch size mismatch across encoder/decoder inputs.")
        if x_enc.shape[1] != self.seq_len:
            raise ValueError(f"x_enc time length must equal seq_len={self.seq_len}, got {x_enc.shape[1]}.")
        if x_mark_enc.shape[1] != self.seq_len:
            raise ValueError(f"x_mark_enc time length must equal seq_len={self.seq_len}, got {x_mark_enc.shape[1]}.")
        if x_enc.shape[2] != self.configs.enc_in:
            raise ValueError(f"x_enc feature length must equal enc_in={self.configs.enc_in}, got {x_enc.shape[2]}.")
        expected_dec_len = self.label_len + self.pred_len
        if x_dec.shape[1] != expected_dec_len or x_mark_dec.shape[1] != expected_dec_len:
            raise ValueError(
                f"Decoder lengths must equal label_len+pred_len={expected_dec_len}, got x_dec={x_dec.shape[1]}, x_mark_dec={x_mark_dec.shape[1]}."
            )
        if x_dec.shape[2] != self.configs.c_out and x_dec.shape[2] != self.configs.enc_in:
            raise ValueError(f"x_dec feature length must equal c_out={self.configs.c_out} or enc_in={self.configs.enc_in}, got {x_dec.shape[2]}.")
        
        if not getattr(self.configs, 'tft_allow_custom_known', False):
            if x_mark_enc.shape[2] != self.known_len:
                raise ValueError(f"x_mark_enc feature length must equal known_len={self.known_len}, got {x_mark_enc.shape[2]}.")
            if x_mark_dec.shape[2] != self.known_len:
                raise ValueError(f"x_mark_dec feature length must equal known_len={self.known_len}, got {x_mark_dec.shape[2]}.")
        else:
            if x_mark_enc.shape[2] != self.known_len or x_mark_dec.shape[2] != self.known_len:
                raise ValueError(
                    f"With tft_allow_custom_known=True, x_mark_* feature length must equal tft_known_len={self.known_len}. "
                    f"Got x_mark_enc={x_mark_enc.shape[2]}, x_mark_dec={x_mark_dec.shape[2]}."
                )
                
        if self.debug_checks and (
            not torch.isfinite(x_enc).all()
            or not torch.isfinite(x_mark_enc).all()
            or not torch.isfinite(x_dec).all()
            or not torch.isfinite(x_mark_dec).all()
        ):
            raise ValueError("TFT inputs contain NaN/Inf values.")

    @staticmethod
    def _split_vsn_weight_payload(weight_payload):
        if isinstance(weight_payload, dict):
            return weight_payload.get('selection'), weight_payload.get('graph_attention')
        return weight_payload, None

    @staticmethod
    def _split_static_vsn_weight_payload(weight_payload):
        if not isinstance(weight_payload, dict):
            return weight_payload, None
        static_weights = {}
        static_graph_attention = {}
        for context_key, context_payload in weight_payload.items():
            if isinstance(context_payload, dict):
                static_weights[context_key] = context_payload.get('selection')
                static_graph_attention[context_key] = context_payload.get('graph_attention')
            else:
                static_weights[context_key] = context_payload
                static_graph_attention[context_key] = None
        if not static_weights:
            static_weights = None
        if not static_graph_attention:
            static_graph_attention = None
        return static_weights, static_graph_attention

    @staticmethod
    def _ordered_quantile_projection(raw_quantile_out):
        base = raw_quantile_out[:, :, :1, :]
        if raw_quantile_out.shape[2] == 1:
            return base
        deltas = F.softplus(raw_quantile_out[:, :, 1:, :])
        ordered_tail = base + torch.cumsum(deltas, dim=2)
        return torch.cat([base, ordered_tail], dim=2)

    def _extract_static_values(self, raw_x_enc):
        if self.static_len == 0:
            return None

        static_series = raw_x_enc.index_select(
            dim=-1,
            index=raw_x_enc.new_tensor(self.resolved_tft_schema.static_positions, dtype=torch.long),
        )
        static_reference = static_series[:, :1, :]
        max_static_drift = (static_series - static_reference).abs().amax(dim=1)
        tolerance = float(getattr(self.configs, 'tft_static_const_tol', 1e-6))
        if torch.any(max_static_drift > tolerance):
            offending = torch.nonzero(max_static_drift > tolerance, as_tuple=False)[0]
            batch_idx = int(offending[0].item())
            static_idx = int(offending[1].item())
            drift_value = float(max_static_drift[batch_idx, static_idx].item())
            raise ValueError(
                "Declared static TFT features must remain constant across encoder time. "
                f"Found drift {drift_value:.6g} at batch index {batch_idx}, static channel {static_idx}, "
                f"tolerance={tolerance:.6g}."
            )
        return static_reference.squeeze(1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation: bool = False):
        self.last_quantile_predictions = None
        raw_x_enc = x_enc
        static_values = self._extract_static_values(raw_x_enc)
        # Normalization from Non-stationary Transformer
        var = torch.var(x_enc, dim=1, keepdim=True, unbiased=False)
        if (var < 1e-8).any():
            warnings.warn("Near-constant channels detected; normalization may amplify noise.")
        if self.use_revin:
            x_enc = self.revin(x_enc, 'norm')
            means = None
            stdev = None
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.clamp(var, min=1e-10) + 1e-5)
            x_enc = x_enc / stdev

        # Data embedding
        # static_input: [B,C,d], observed_input:[B,T,C,d], known_input: [B,T,C,d]
        static_input, observed_input, known_input = self.embedding(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            static_values=static_values,
        )

        # Static context
        # c_s,...,c_e: [B,d]
        if return_interpretation:
            static_contexts, static_weight_payload = self.static_encoder(static_input, return_weights=True)
            c_s, c_c, c_h, c_e = static_contexts
            static_weights, static_graph_attention = self._split_static_vsn_weight_payload(static_weight_payload)
        else:
            c_s, c_c, c_h, c_e = self.static_encoder(static_input)

        # Temporal input Selection
        history_input = torch.cat([observed_input, known_input[:,:self.seq_len]], dim=-2)
        future_input = known_input[:,self.seq_len:]
        # Capture pre-VSN covariate embeddings for covariate-aware enrichment (Phase C)
        if self.use_covariate_reattention:
            C_hist = history_input.shape[-2]
            C_fut = future_input.shape[-2]
            if C_fut < C_hist:
                pad = torch.zeros(*future_input.shape[:-2], C_hist - C_fut, future_input.shape[-1],
                                  device=future_input.device, dtype=future_input.dtype)
                future_padded = torch.cat([future_input, pad], dim=-2)
            else:
                future_padded = future_input
            pre_vsn_embs = torch.cat([history_input.detach(), future_padded.detach()], dim=1)
        else:
            pre_vsn_embs = None
        if return_interpretation:
            history_input, history_weight_payload = self.history_vsn(history_input, c_s, return_weights=True)
            future_input, future_weight_payload = self.future_vsn(future_input, c_s, return_weights=True)
            history_weights, history_graph_attention = self._split_vsn_weight_payload(history_weight_payload)
            future_weights, future_graph_attention = self._split_vsn_weight_payload(future_weight_payload)
        else:
            history_input = self.history_vsn(history_input, c_s)
            future_input = self.future_vsn(future_input, c_s)

        # TFT main procedure after variable selection
        # history_input: [B,T,d], future_input: [B,T,d]
        if self.use_quantile_head:
            if return_interpretation:
                dec_out, attention_weights, decoder_hidden = self.temporal_fusion_decoder(
                    history_input,
                    future_input,
                    c_c,
                    c_h,
                    c_e,
                    return_attention=True,
                    return_decoder_hidden=True,
                    pre_vsn_embs=pre_vsn_embs,
                )
            else:
                dec_out, decoder_hidden = self.temporal_fusion_decoder(
                    history_input,
                    future_input,
                    c_c,
                    c_h,
                    c_e,
                    return_decoder_hidden=True,
                    pre_vsn_embs=pre_vsn_embs,
                )
        else:
            decoder_hidden = None
            if return_interpretation:
                dec_out, attention_weights = self.temporal_fusion_decoder(
                    history_input,
                    future_input,
                    c_c,
                    c_h,
                    c_e,
                    return_attention=True,
                    pre_vsn_embs=pre_vsn_embs,
                )
            else:
                dec_out = self.temporal_fusion_decoder(history_input, future_input, c_c, c_h, c_e, pre_vsn_embs=pre_vsn_embs)
        self.last_moe_aux_loss = self.temporal_fusion_decoder.last_moe_aux_loss
        self.last_moe_importance_sum = self.temporal_fusion_decoder.last_moe_importance_sum
        self.last_moe_load_sum = self.temporal_fusion_decoder.last_moe_load_sum
        self.last_moe_token_count = self.temporal_fusion_decoder.last_moe_token_count
        if self.use_quantile_head:
            quantile_out = self.quantile_projection(decoder_hidden)
            quantile_out = quantile_out.view(decoder_hidden.shape[0], self.pred_len, len(self.quantiles), self.configs.c_out)
            quantile_out = self._ordered_quantile_projection(quantile_out)
        else:
            quantile_out = None

        # De-Normalization from Non-stationary Transformer
        target_pos = self.target_pos_buf
        if self.use_revin:
            if dec_out is not None:
                full_dec = torch.zeros(
                    dec_out.shape[0],
                    self.pred_len,
                    self.configs.enc_in,
                    device=dec_out.device,
                    dtype=dec_out.dtype,
                )
                full_dec.scatter_(
                    -1,
                    target_pos.view(1, 1, -1).expand(dec_out.shape[0], self.pred_len, -1),
                    dec_out,
                )
                full_dec = self.revin(full_dec, 'denorm')
                dec_out = full_dec.index_select(-1, target_pos)
            if quantile_out is not None:
                full_quantile = torch.zeros(
                    quantile_out.shape[0],
                    self.pred_len,
                    quantile_out.shape[2],
                    self.configs.enc_in,
                    device=quantile_out.device,
                    dtype=quantile_out.dtype,
                )
                full_quantile.scatter_(
                    -1,
                    target_pos.view(1, 1, 1, -1).expand(quantile_out.shape[0], self.pred_len, quantile_out.shape[2], -1),
                    quantile_out,
                )
                full_quantile = full_quantile.reshape(full_quantile.shape[0], -1, full_quantile.shape[-1])
                full_quantile = self.revin(full_quantile, 'denorm')
                full_quantile = full_quantile.view(quantile_out.shape[0], self.pred_len, quantile_out.shape[2], self.configs.enc_in)
                quantile_out = full_quantile.index_select(-1, target_pos)
        else:
            target_stdev = stdev[:, 0, :].index_select(-1, target_pos).unsqueeze(1).expand(-1, self.pred_len, -1)
            target_means = means[:, 0, :].index_select(-1, target_pos).unsqueeze(1).expand(-1, self.pred_len, -1)
            if dec_out is not None:
                dec_out = dec_out * target_stdev
                dec_out = dec_out + target_means
            if quantile_out is not None:
                quantile_out = quantile_out * target_stdev.unsqueeze(2)
                quantile_out = quantile_out + target_means.unsqueeze(2)
        self.last_quantile_predictions = quantile_out
        if self.output_mode == 'quantile':
            dec_out = quantile_out[:, :, self.quantile_median_index, :]
        if return_interpretation:
            attention_weights_full = None
            attention_fusion_alpha = None
            lag_attention_weights = None
            lag_scale_weights = None
            attention_branch_weights = None
            cross_attention_weights = None
            interaction_contribution = None
            interaction_gates = None
            expert_routing = None
            regime_probabilities = None
            regime_probabilities_pooled = None
            quantile_predictions = self.last_quantile_predictions
            moe_aux_loss = self.last_moe_aux_loss.detach() if torch.is_tensor(self.last_moe_aux_loss) else self.last_moe_aux_loss
            decoder_layer_payloads = None
            decoder_num_layers = None
            position_bias_type = self.position_bias_type
            temporal_backbone_type = self.temporal_backbone_type
            attention_backend_config = self.attention_backend
            attention_backend_used = None
            cross_attention_backend_used = None
            fft_gate_mean = None
            fft_learned_mask_mean = None
            fft_learned_mask_std = None
            fft_learned_mask_peak_bin_mean = None
            temporal_positions = None
            tc_active = False
            tc_compressed_history_len = None
            if isinstance(attention_weights, dict):
                attention_weights_full = attention_weights.get('full')
                attention_fusion_alpha = attention_weights.get('fusion_alpha')
                lag_attention_weights = attention_weights.get('lag_attention')
                lag_scale_weights = attention_weights.get('lag_scale_weights')
                attention_branch_weights = attention_weights.get('attention_branch_weights')
                cross_attention_weights = attention_weights.get('cross_attention')
                interaction_contribution = attention_weights.get('interaction_contribution')
                interaction_gates = attention_weights.get('interaction_gates')
                expert_routing = attention_weights.get('expert_routing')
                regime_probabilities = attention_weights.get('regime_probabilities')
                regime_probabilities_pooled = attention_weights.get('regime_probabilities_pooled')
                moe_aux_loss = attention_weights.get('moe_aux_loss', moe_aux_loss)
                decoder_layer_payloads = attention_weights.get('decoder_layer_payloads')
                decoder_num_layers = attention_weights.get('decoder_num_layers')
                position_bias_type = attention_weights.get('position_bias_type', position_bias_type)
                temporal_backbone_type = attention_weights.get('temporal_backbone_type', temporal_backbone_type)
                attention_backend_config = attention_weights.get('attention_backend_config', attention_backend_config)
                attention_backend_used = attention_weights.get('attention_backend_used')
                cross_attention_backend_used = attention_weights.get('cross_attention_backend_used')
                fft_gate_mean = attention_weights.get('fft_gate_mean')
                fft_learned_mask_mean = attention_weights.get('fft_learned_mask_mean')
                fft_learned_mask_std = attention_weights.get('fft_learned_mask_std')
                fft_learned_mask_peak_bin_mean = attention_weights.get('fft_learned_mask_peak_bin_mean')
                temporal_positions = attention_weights.get('temporal_positions')
                tc_active = attention_weights.get('tc_active', False)
                tc_compressed_history_len = attention_weights.get('tc_compressed_history_len')
                attention_weights = attention_weights.get('interpretable')
            return {
                'predictions': dec_out,
                'attention_weights': attention_weights,
                'attention_weights_full': attention_weights_full,
                'attention_fusion_alpha': attention_fusion_alpha,
                'attention_branch_weights': attention_branch_weights,
                'cross_attention_weights': cross_attention_weights,
                'lag_attention_weights': lag_attention_weights,
                'lag_scale_weights': lag_scale_weights,
                'interaction_contribution': interaction_contribution,
                'interaction_gates': interaction_gates,
                'expert_routing': expert_routing,
                'regime_probabilities': regime_probabilities,
                'regime_probabilities_pooled': regime_probabilities_pooled,
                'quantile_predictions': quantile_predictions,
                'quantiles': self.quantiles,
                'output_mode': self.output_mode,
                'moe_aux_loss': moe_aux_loss,
                'decoder_layer_payloads': decoder_layer_payloads,
                'decoder_num_layers': decoder_num_layers,
                'position_bias_type': position_bias_type,
                'temporal_backbone_type': temporal_backbone_type,
                'attention_backend_config': attention_backend_config,
                'attention_backend_used': attention_backend_used,
                'cross_attention_backend_used': cross_attention_backend_used,
                'fft_gate_mean': fft_gate_mean,
                'fft_learned_mask_mean': fft_learned_mask_mean,
                'fft_learned_mask_std': fft_learned_mask_std,
                'fft_learned_mask_peak_bin_mean': fft_learned_mask_peak_bin_mean,
                'temporal_positions': temporal_positions,
                'tc_active': tc_active,
                'tc_compressed_history_len': tc_compressed_history_len,
                'history_vsn_weights': history_weights,
                'history_graph_attention': history_graph_attention,
                'future_vsn_weights': future_weights,
                'future_graph_attention': future_graph_attention,
                'static_vsn_weights': static_weights,
                'static_graph_attention': static_graph_attention,
                'observed_feature_names': tuple(
                    self.resolved_tft_schema.feature_names[idx] for idx in self.resolved_tft_schema.observed_positions
                ),
                'known_feature_names': tuple(self.resolved_tft_schema.known_feature_names),
                'history_feature_names': tuple(
                    [self.resolved_tft_schema.feature_names[idx] for idx in self.resolved_tft_schema.observed_positions]
                    + list(self.resolved_tft_schema.known_feature_names)
                ),
                'future_feature_names': tuple(self.resolved_tft_schema.known_feature_names),
                'static_feature_names': tuple(
                    self.resolved_tft_schema.feature_names[idx] for idx in self.resolved_tft_schema.static_positions
                ),
                'interpretation_flags': {
                    'is_canonical_vsn_attribution': (
                        self.is_canonical_profile
                        and not getattr(self.configs, 'tft_cross_variable_mixing', False)
                        and not getattr(self.configs, 'tft_vsn_residual_bypass', False)
                        and not getattr(self.configs, 'tft_full_attention', False)
                        and not getattr(self.configs, 'tft_dual_attention_fusion', False)
                        and not getattr(self.configs, 'tft_use_fft_branch', False)
                        and not getattr(self.configs, 'tft_use_higher_order', False)
                        and not getattr(self.configs, 'tft_covariate_reattention', False)
                    ),
                    'uses_graph_pre_mixing': bool(getattr(self.configs, 'tft_cross_variable_mixing', False)),
                    'uses_vsn_bypass': bool(getattr(self.configs, 'tft_vsn_residual_bypass', False)),
                    'uses_noninterpretable_attention_branch': bool(
                        getattr(self.configs, 'tft_full_attention', False)
                        or getattr(self.configs, 'tft_dual_attention_fusion', False)
                        or getattr(self.configs, 'tft_use_explicit_cross_attention', False)
                    ),
                    'uses_global_spectral_mixing': bool(getattr(self.configs, 'tft_use_fft_branch', False)),
                    'routing_is_detached': expert_routing is not None,
                },
                'static_context': {'c_s': c_s, 'c_c': c_c, 'c_h': c_h, 'c_e': c_e},
                'use_revin': self.use_revin,
            }
        return dec_out

    def _make_structured_output(self, point_forecast: torch.Tensor, history_len: int) -> TFTForecastOutput:
        point_full = torch.cat([
            torch.zeros(
                point_forecast.shape[0],
                history_len,
                self.configs.c_out,
                device=point_forecast.device,
                dtype=point_forecast.dtype,
            ),
            point_forecast,
        ], dim=1)
        return TFTForecastOutput(
            point_forecast=point_forecast,
            point_full=point_full,
            quantile_forecast=self.last_quantile_predictions,
            moe_importance_sum=self.last_moe_importance_sum,
            moe_load_sum=self.last_moe_load_sum,
            moe_token_count=self.last_moe_token_count,
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation: bool = False, return_auxiliary: bool = False):
        self._validate_inputs(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.task_name == 'long_term_forecast':
            if return_interpretation:
                payload = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation=True)
                dec_out = payload['predictions']
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B,pred_len,C]
            structured_output = self._make_structured_output(dec_out, x_enc.shape[1])
            dec_out = structured_output.point_full
            if return_interpretation:
                payload['predictions_full'] = dec_out
                return payload
            if return_auxiliary:
                return structured_output
            return dec_out  # [B, T, D]
        return None
