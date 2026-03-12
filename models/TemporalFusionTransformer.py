import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, TemporalEmbedding
from torch import Tensor
from typing import Optional
from collections import namedtuple
import warnings

# static: time-independent features
# observed: time features of the past(e.g. predicted targets)
# known: known information about the past and future(i.e. time stamp)
TypePos = namedtuple('TypePos', ['static', 'observed'])

# When you want to use new dataset, please add the index of 'static, observed' columns here.
# 'known' columns needn't be added, because 'known' inputs are automatically judged and provided by the program.
datatype_dict = {
    'ETTh1': TypePos([], [x for x in range(7)]),
    'ETTh2': TypePos([], [x for x in range(7)]),
    'ETTm1': TypePos([], [x for x in range(7)]),
    'ETTm2': TypePos([], [x for x in range(7)]),
}


def get_typepos(configs) -> TypePos:
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
    if not hasattr(configs, 'enc_in') or not hasattr(configs, 'c_out'):
        raise KeyError("configs must define enc_in and c_out for TFT target mapping.")

    target_pos = getattr(configs, 'tft_target_pos', None)
    if target_pos is None:
        if configs.c_out == configs.enc_in:
            return [x for x in range(configs.c_out)]
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
    if embed_type != 'timeF':
        if freq == 't':
            return 5
        else:
            return 4
    else:
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        return freq_map[freq]


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


class TFTEmbedding(nn.Module):
    def __init__(self, configs):
        super(TFTEmbedding, self).__init__()
        self.pred_len = configs.pred_len
        typepos = get_typepos(configs)
        self.static_pos = typepos.static
        self.observed_pos = typepos.observed
        self.static_len = len(self.static_pos)
        self.observed_len = len(self.observed_pos)

        self.static_embedding = nn.ModuleList([DataEmbedding(1,configs.d_model,dropout=configs.dropout) for _ in range(self.static_len)]) \
            if self.static_len else None
        self.observed_embedding = nn.ModuleList([DataEmbedding(1,configs.d_model,dropout=configs.dropout) for _ in range(self.observed_len)])
        self.known_embedding = TFTTemporalEmbedding(configs.d_model, configs.embed, configs.freq) \
            if configs.embed != 'timeF' else TFTTimeFeatureEmbedding(configs.d_model, configs.embed, configs.freq)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.static_len:
            # static_input: [B,C,d_model]
            static_input = torch.stack([embed(x_enc[:,:1,self.static_pos[i]].unsqueeze(-1), None).squeeze(1) for i, embed in enumerate(self.static_embedding)], dim=-2)
        else:
            static_input = None

        # observed_input: [B,T,C,d_model]
        observed_input = torch.stack([embed(x_enc[:,:,self.observed_pos[i]].unsqueeze(-1), None) for i, embed in enumerate(self.observed_embedding)], dim=-2)

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


class CrossVariableAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super(CrossVariableAttention, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B,T,C,d] or [B,C,d]
        if x.ndim == 4:
            B, T, C, d = x.shape
            x_flat = x.reshape(B * T, C, d)
            attn_out, _ = self.mha(x_flat, x_flat, x_flat, need_weights=False)
            out = self.layer_norm(x_flat + attn_out)
            return out.reshape(B, T, C, d)
        elif x.ndim == 3:
            attn_out, _ = self.mha(x, x, x, need_weights=False)
            return self.layer_norm(x + attn_out)
        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(self, d_model, variable_num, dropout=0.0, use_swiglu=False, cross_variable_mixing=False, n_heads=4):
        super(VariableSelectionNetwork, self).__init__()
        self.cross_mixing = CrossVariableAttention(d_model, n_heads, dropout) if cross_variable_mixing else None
        self.joint_grn = GRN(d_model * variable_num, variable_num, hidden_size=d_model, context_size=d_model, dropout=dropout, use_swiglu=use_swiglu)
        self.variable_grns = nn.ModuleList([GRN(d_model, d_model, dropout=dropout, use_swiglu=use_swiglu) for _ in range(variable_num)])

    def forward(self, x: Tensor, context: Optional[Tensor] = None, return_weights: bool = False):
        if self.cross_mixing is not None:
            x = self.cross_mixing(x)
        # x: [B,T,C,d] or [B,C,d]
        # selection_weights: [B,T,C] or [B,C]
        # x_processed: [B,T,d,C] or [B,d,C]
        # selection_result: [B,T,d] or [B,d]
        x_flattened = torch.flatten(x, start_dim=-2)
        selection_weights = self.joint_grn(x_flattened, context)
        selection_weights = F.softmax(selection_weights, dim=-1)

        x_processed = torch.stack([grn(x[...,i,:]) for i, grn in enumerate(self.variable_grns)], dim=-1)

        selection_result = torch.matmul(x_processed, selection_weights.unsqueeze(-1)).squeeze(-1)
        if return_weights:
            return selection_result, selection_weights
        return selection_result


class StaticCovariateEncoder(nn.Module):
    def __init__(self, d_model, static_len, dropout=0.0, use_swiglu=False, cross_variable_mixing=False, n_heads=4):
        super(StaticCovariateEncoder, self).__init__()
        self.static_vsn = VariableSelectionNetwork(d_model, static_len, dropout=dropout, use_swiglu=use_swiglu, cross_variable_mixing=cross_variable_mixing, n_heads=n_heads) if static_len else None
        self.grns = nn.ModuleList([GRN(d_model, d_model, dropout=dropout, use_swiglu=use_swiglu) for _ in range(4)])

    def forward(self, static_input, return_weights: bool = False):
        # static_input: [B,C,d]
        if static_input is not None:
            if return_weights:
                static_features, static_weights = self.static_vsn(static_input, return_weights=True)
                return [grn(static_features) for grn in self.grns], static_weights
            static_features = self.static_vsn(static_input)
            return [grn(static_features) for grn in self.grns]
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
        self.qkv_linears = nn.Linear(configs.d_model, (2 * self.n_heads + 1) * self.d_head, bias=False)
        self.out_projection = nn.Linear(self.d_head, configs.d_model, bias=False)
        self.out_dropout = nn.Dropout(configs.dropout)
        self.scale = self.d_head ** -0.5
    def _causal_mask(self, seq_len: int, device, dtype):
        return torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype), 1)

    def forward(self, x, return_attention: bool = False):
        # Q,K,V are all from x
        B, T, d_model = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_heads * self.d_head, self.n_heads * self.d_head, self.d_head), dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.d_head)

        attention_score = torch.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))  # [B,n,T,T]
        attention_score.mul_(self.scale)
        attention_score = torch.clamp(attention_score, min=-1e4, max=1e4)
        if not torch.isfinite(attention_score).all():
            raise ValueError("Unmasked attention scores contain NaN/Inf values.")
        attention_score = attention_score + self._causal_mask(T, attention_score.device, attention_score.dtype)
        attention_prob = F.softmax(attention_score, dim=3)  # [B,n,T,T]
        if not torch.isfinite(attention_prob).all():
            raise ValueError("Attention probabilities contain NaN/Inf values.")

        attention_out = torch.matmul(attention_prob, v.unsqueeze(1))  # [B,n,T,d]
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
        self.full_attention = getattr(configs, 'tft_full_attention', False)

        self.history_encoder = nn.LSTM(configs.d_model, configs.d_model, batch_first=True)
        self.future_encoder = nn.LSTM(configs.d_model, configs.d_model, batch_first=True)
        self.gate_after_lstm = GateAddNorm(configs.d_model, configs.d_model, use_swiglu=self.use_swiglu)
        self.enrichment_grn = GRN(configs.d_model, configs.d_model, context_size=configs.d_model, dropout=configs.dropout, use_swiglu=self.use_swiglu)
        if self.full_attention:
            self.attention = nn.MultiheadAttention(configs.d_model, configs.n_heads, dropout=configs.dropout, batch_first=True)
        else:
            self.attention = InterpretableMultiHeadAttention(configs)
        self.gate_after_attention = GateAddNorm(configs.d_model, configs.d_model, use_swiglu=self.use_swiglu)
        self.position_wise_grn = GRN(configs.d_model, configs.d_model, dropout=configs.dropout, use_swiglu=self.use_swiglu)
        self.gate_final = GateAddNorm(configs.d_model, configs.d_model, use_swiglu=self.use_swiglu)

    def _causal_mask(self, seq_len: int, device, dtype):
        return torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype), 1)

    def forward(self, history_input, future_input, c_c, c_h, c_e, return_attention: bool = False):
        c = (c_c.unsqueeze(0), c_h.unsqueeze(0)) if c_c is not None and c_h is not None else None
        historical_features, state = self.history_encoder(history_input, c)
        future_features, _ = self.future_encoder(future_input, state)

        temporal_input = torch.cat([history_input, future_input], dim=1)
        temporal_features = torch.cat([historical_features, future_features], dim=1)
        temporal_features = self.gate_after_lstm(temporal_features, temporal_input)

        enriched_features = self.enrichment_grn(temporal_features, c_e)

        if self.full_attention:
            # Match standard TFT causal masking
            seq_len = enriched_features.shape[1]
            attn_mask = self._causal_mask(seq_len, enriched_features.device, enriched_features.dtype)
            
            # average_attn_weights=False ensures we get [B, n_heads, T, T] to match Interpretable API
            attention_out, attention_prob = self.attention(
                enriched_features, enriched_features, enriched_features, 
                need_weights=return_attention, attn_mask=attn_mask, average_attn_weights=False
            )
        else:
            if return_attention:
                attention_out, attention_prob = self.attention(enriched_features, return_attention=True)
            else:
                attention_out = self.attention(enriched_features)
                attention_prob = None

        attention_out = self.gate_after_attention(attention_out, enriched_features)
        out = self.position_wise_grn(attention_out)
        out = self.gate_final(out, temporal_features)
        if return_attention:
            return out, attention_prob
        return out


class TemporalFusionDecoder(nn.Module):
    def __init__(self, configs):
        super(TemporalFusionDecoder, self).__init__()
        self.e_layers = getattr(configs, 'e_layers', 1)
        self.pred_len = configs.pred_len
        self.layers = nn.ModuleList([TemporalFusionDecoderLayer(configs) for _ in range(self.e_layers)])
        self.out_projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, history_input, future_input, c_c, c_h, c_e, return_attention: bool = False):
        attention_probs = []
        curr_history = history_input
        curr_future = future_input
        
        for layer in self.layers:
            if return_attention:
                out, attention_prob = layer(curr_history, curr_future, c_c, c_h, c_e, return_attention=True)
                attention_probs.append(attention_prob)
            else:
                out = layer(curr_history, curr_future, c_c, c_h, c_e)
            curr_history = out[:, :history_input.shape[1], :]
            curr_future = out[:, history_input.shape[1]:, :]
        
        dec_out = out[:, -self.pred_len:, :]
        projected = self.out_projection(dec_out)
        
        if return_attention:
            return projected, attention_probs[-1] if attention_probs else None
        return projected


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Number of variables
        typepos = get_typepos(configs)
        self.static_len = len(typepos.static)
        self.observed_len = len(typepos.observed)
        self.target_pos = get_target_pos(configs)
        self.known_len = get_known_len(configs.embed, configs.freq)

        self.embedding = TFTEmbedding(configs)
        
        self.use_swiglu = getattr(configs, 'tft_use_swiglu', False)
        self.cross_mix = getattr(configs, 'tft_cross_variable_mixing', False)
        self.n_heads = getattr(configs, 'n_heads', 4)

        self.static_encoder = StaticCovariateEncoder(
            configs.d_model, self.static_len, dropout=configs.dropout, 
            use_swiglu=self.use_swiglu, cross_variable_mixing=self.cross_mix, n_heads=self.n_heads
        )
        self.history_vsn = VariableSelectionNetwork(
            configs.d_model, self.observed_len + self.known_len, dropout=configs.dropout,
            use_swiglu=self.use_swiglu, cross_variable_mixing=self.cross_mix, n_heads=self.n_heads
        )
        self.future_vsn = VariableSelectionNetwork(
            configs.d_model, self.known_len, dropout=configs.dropout,
            use_swiglu=self.use_swiglu, cross_variable_mixing=self.cross_mix, n_heads=self.n_heads
        )
        self.temporal_fusion_decoder = TemporalFusionDecoder(configs)

    def _validate_inputs(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if x_enc.ndim != 3 or x_mark_enc.ndim != 3 or x_dec.ndim != 3 or x_mark_dec.ndim != 3:
            raise ValueError("All TFT inputs must be rank-3 tensors [B,T,D].")
        if x_enc.shape[0] != x_mark_enc.shape[0] or x_enc.shape[0] != x_dec.shape[0] or x_enc.shape[0] != x_mark_dec.shape[0]:
            raise ValueError("Batch size mismatch across encoder/decoder inputs.")
        if x_enc.shape[1] != self.seq_len:
            raise ValueError(f"x_enc time length must equal seq_len={self.seq_len}, got {x_enc.shape[1]}.")
        if x_enc.shape[2] != self.configs.enc_in:
            raise ValueError(f"x_enc feature length must equal enc_in={self.configs.enc_in}, got {x_enc.shape[2]}.")
        expected_dec_len = self.label_len + self.pred_len
        if x_dec.shape[1] != expected_dec_len or x_mark_dec.shape[1] != expected_dec_len:
            raise ValueError(
                f"Decoder lengths must equal label_len+pred_len={expected_dec_len}, got x_dec={x_dec.shape[1]}, x_mark_dec={x_mark_dec.shape[1]}."
            )
        if x_dec.shape[2] != self.configs.c_out:
            raise ValueError(f"x_dec feature length must equal c_out={self.configs.c_out}, got {x_dec.shape[2]}.")
        
        if not getattr(self.configs, 'tft_allow_custom_known', False):
            if x_mark_enc.shape[2] != self.known_len:
                raise ValueError(f"x_mark_enc feature length must equal known_len={self.known_len}, got {x_mark_enc.shape[2]}.")
            if x_mark_dec.shape[2] != self.known_len:
                raise ValueError(f"x_mark_dec feature length must equal known_len={self.known_len}, got {x_mark_dec.shape[2]}.")
                
        if not torch.isfinite(x_enc).all() or not torch.isfinite(x_mark_enc).all() or not torch.isfinite(x_dec).all() or not torch.isfinite(x_mark_dec).all():
            raise ValueError("TFT inputs contain NaN/Inf values.")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation: bool = False):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        var = torch.var(x_enc, dim=1, keepdim=True, unbiased=False)
        if (var < 1e-8).any():
            warnings.warn("Near-constant channels detected; normalization may amplify noise.")
        stdev = torch.sqrt(torch.clamp(var, min=1e-10) + 1e-5)
        x_enc /= stdev

        # Data embedding
        # static_input: [B,C,d], observed_input:[B,T,C,d], known_input: [B,T,C,d]
        static_input, observed_input, known_input = self.embedding(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # Static context
        # c_s,...,c_e: [B,d]
        if return_interpretation:
            static_contexts, static_weights = self.static_encoder(static_input, return_weights=True)
            c_s, c_c, c_h, c_e = static_contexts
        else:
            c_s, c_c, c_h, c_e = self.static_encoder(static_input)

        # Temporal input Selection
        history_input = torch.cat([observed_input, known_input[:,:self.seq_len]], dim=-2)
        future_input = known_input[:,self.seq_len:]
        if return_interpretation:
            history_input, history_weights = self.history_vsn(history_input, c_s, return_weights=True)
            future_input, future_weights = self.future_vsn(future_input, c_s, return_weights=True)
        else:
            history_input = self.history_vsn(history_input, c_s)
            future_input = self.future_vsn(future_input, c_s)

        # TFT main procedure after variable selection
        # history_input: [B,T,d], future_input: [B,T,d]
        if return_interpretation:
            dec_out, attention_weights = self.temporal_fusion_decoder(
                history_input,
                future_input,
                c_c,
                c_h,
                c_e,
                return_attention=True,
            )
        else:
            dec_out = self.temporal_fusion_decoder(history_input, future_input, c_c, c_h, c_e)

        # De-Normalization from Non-stationary Transformer
        target_pos = torch.as_tensor(self.target_pos, device=x_enc.device, dtype=torch.long)
        target_stdev = stdev[:, 0, :].index_select(-1, target_pos).unsqueeze(1).repeat(1, self.pred_len, 1)
        target_means = means[:, 0, :].index_select(-1, target_pos).unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out * target_stdev
        dec_out = dec_out + target_means
        if return_interpretation:
            return {
                'predictions': dec_out,
                'attention_weights': attention_weights,
                'history_vsn_weights': history_weights,
                'future_vsn_weights': future_weights,
                'static_vsn_weights': static_weights,
                'static_context': {'c_s': c_s, 'c_c': c_c, 'c_h': c_h, 'c_e': c_e},
            }
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation: bool = False):
        self._validate_inputs(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if return_interpretation:
                payload = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation=True)
                dec_out = payload['predictions']
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B,pred_len,C]
            history_pad = torch.zeros(
                x_enc.shape[0],
                x_enc.shape[1],
                self.configs.c_out,
                device=x_enc.device,
                dtype=x_enc.dtype,
            )
            dec_out = torch.cat([history_pad, dec_out], dim=1)
            if return_interpretation:
                payload['predictions_full'] = dec_out
                return payload
            return dec_out  # [B, T, D]
        return None