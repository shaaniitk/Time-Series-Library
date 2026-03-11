import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from neuralforecast.models.tft import TFT, GRN, VariableSelectionNetwork, TFTEmbedding, InterpretableMultiHeadAttention
from utils.timefeatures import time_features

class EnhancedGRN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, context_hidden_size=None, dropout=0):
        super().__init__()
        # Initialize Nixtla's GRN
        self.grn = GRN(input_size, hidden_size, output_size, context_hidden_size, dropout)
        # Upgrade to SwiGLU (SiLU * Gated Linear)
        self.grn.glu = nn.Linear(hidden_size, (output_size if output_size else hidden_size) * 2)

    def forward(self, a, c: Optional[torch.Tensor] = None):
        x = self.grn.lin_a(a)
        if c is not None:
            c_proj = self.grn.lin_c(c)
            if c_proj.ndim == 2:
                c_proj = c_proj.unsqueeze(1)
            x = x + c_proj
        x = F.elu(x)
        x = self.grn.lin_i(x)
        x = self.grn.dropout(x)
        
        # SwiGLU instead of standard GLU
        glu_out = self.grn.glu(x)
        out_size = glu_out.shape[-1] // 2
        x_val, x_gate = glu_out[..., :out_size], glu_out[..., out_size:]
        x = F.silu(x_val) * x_gate
        
        y = a if not self.grn.out_proj else self.grn.out_proj(a)
        x = x + y
        x = self.grn.layer_norm(x)
        return x

class EnhancedVSN(nn.Module):
    def __init__(self, hidden_size, num_inputs, dropout):
        super().__init__()
        # VSN Initialization with version compatibility
        try:
            self.vsn = VariableSelectionNetwork(hidden_size, num_inputs, dropout)
        except TypeError:
            self.vsn = VariableSelectionNetwork(hidden_size, num_inputs, dropout, None)
        
        # Upgrade the sub-GRNs inside VSN with our EnhancedGRN
        self.vsn.joint_grn = EnhancedGRN(
            input_size=hidden_size * num_inputs,
            hidden_size=hidden_size,
            output_size=num_inputs,
            context_hidden_size=hidden_size,
        )
        self.vsn.var_grns = nn.ModuleList([
            EnhancedGRN(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)
            for _ in range(num_inputs)
        ])
        
        # Adds Residual Bypass 
        self.residual_projection = nn.Linear(hidden_size * num_inputs, hidden_size)
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        Xi = x.reshape(*x.shape[:-2], -1)
        grn_outputs = self.vsn.joint_grn(Xi, c=context)
        sparse_weights = F.softmax(grn_outputs, dim=-1)
        transformed_embed_list = [m(x[..., i, :]) for i, m in enumerate(self.vsn.var_grns)]
        transformed_embed = torch.stack(transformed_embed_list, dim=-1)
        
        variable_ctx = torch.matmul(transformed_embed, sparse_weights.unsqueeze(-1)).squeeze(-1)
        
        # VSN Residual Bypass Application
        residual = self.residual_projection(Xi)
        variable_ctx = variable_ctx + torch.tanh(self.residual_gate) * residual
        
        return variable_ctx, sparse_weights

class DualInterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head, hidden_size, example_length, attn_dropout, dropout):
        super().__init__()
        self.interpretable_attn = InterpretableMultiHeadAttention(n_head, hidden_size, example_length, attn_dropout, dropout)
        self.full_attn = nn.MultiheadAttention(hidden_size, n_head, dropout=max(dropout, attn_dropout), batch_first=True)
        self.attention_fusion_logit = nn.Parameter(torch.tensor(0.0))

    def _causal_mask(self, seq_len, device, dtype):
        return torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype), diagonal=1)

    def forward(self, x: torch.Tensor, mask_future_timesteps: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # Run Interpretable branch
        interpretable_out, interpretable_weights = self.interpretable_attn(x, mask_future_timesteps)
        
        # Run Full Attention branch
        seq_len = x.shape[1]
        attn_mask = self._causal_mask(seq_len, x.device, x.dtype) if mask_future_timesteps else None
        
        full_out, full_weights = self.full_attn(
            x, x, x, 
            need_weights=True, 
            attn_mask=attn_mask,
            average_attn_weights=False
        )

        # Blend dynamically
        fusion_alpha = torch.sigmoid(self.attention_fusion_logit)
        blended_out = fusion_alpha * full_out + (1.0 - fusion_alpha) * interpretable_out
        
        # Pack weights and alpha into a tuple matching advanced style payload
        return blended_out, (interpretable_weights, full_weights, fusion_alpha)


class ContinuousTFTEmbedding(nn.Module):
    def __init__(self, hidden_size, stat_input_size, futr_input_size, hist_input_size, tgt_size):
        super().__init__()
        self.tft_embed = TFTEmbedding(hidden_size, stat_input_size, futr_input_size, hist_input_size, tgt_size)
        
        # Implement Value Projection and Channel Bias for Custom Embeddings mapping
        self.value_projection = nn.Linear(1, hidden_size, bias=False)
        self.futr_input_size = futr_input_size
        if futr_input_size:
            self.channel_embedding = nn.Embedding(futr_input_size, hidden_size)

    def _apply_continuous_embedding(self, cont: Optional[torch.Tensor], num_channels: int):
        if cont is not None:
            # cont: [B, T, C]
            values = self.value_projection(cont.unsqueeze(-1)) # [B, T, C, H]
            channel_ids = torch.arange(num_channels, device=cont.device)
            channel_bias = self.channel_embedding(channel_ids).unsqueeze(0).unsqueeze(0) # [1, 1, C, H]
            return values + channel_bias
        return None

    def forward(self, target_inp, stat_exog=None, futr_exog=None, hist_exog=None):
        s_inp = None # We ignore static/hist due to Nixtla univariate wrapper usage
        k_inp = self._apply_continuous_embedding(futr_exog, self.futr_input_size)
        o_inp = None 

        while target_inp.ndim < 4:
            target_inp = target_inp.unsqueeze(-1)
        target_inp = torch.matmul(
            target_inp,
            self.tft_embed.tgt_embedding_vectors.unsqueeze(1),
        )
        if target_inp.ndim == 5:
            target_inp = target_inp.squeeze(-2)
        target_inp = target_inp + self.tft_embed.tgt_embedding_bias

        return s_inp, k_inp, o_inp, target_inp


class Model(nn.Module):
    """
    Wrapper for Nixtla's Temporal Fusion Transformer (`neuralforecast>=1.7.0`).
    Incorporates ALL custom TSL enhancements via PyTorch Proxy patching.
    
    NOTE ON MODELING: This wrapper utilizes a strictly univariate backend (Nixtla). 
    To support multivariate payloads [B, T, C], we reshape to [B*C, T]. 
    This flattens channels into the batch dimension (Channel Independence).
    This intrinsically removes any cross-target interaction modeling compared to 
    the native Time-Series-Library TemporalFusionTransformer that processes [B, T, C] jointly.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        from pandas import date_range
        dummy_dates = date_range('2020-01-01', periods=1, freq=configs.freq)
        dummy_feat = time_features(dummy_dates, freq=configs.freq)
        self.time_features_dim = dummy_feat.shape[0]

        self.nixtla_tft = TFT(
            h=self.pred_len,
            input_size=self.seq_len,
            stat_exog_list=None,
            hist_exog_list=None,
            futr_exog_list=[f"futr_{i}" for i in range(self.time_features_dim)],
            hidden_size=configs.d_model,
            n_head=configs.n_heads,
            attn_dropout=configs.dropout,
            dropout=configs.dropout,
            batch_size=configs.batch_size,
            windows_batch_size=configs.batch_size
        )
        
        self.c_out = configs.c_out
        if self.c_out > 1:
            self.cross_channel_mixer = nn.Linear(self.c_out, self.c_out)
        
        # Patching Nixtla with our custom enhancements
        self._inject_enhanced_blocks(self.nixtla_tft)

    def _inject_enhanced_blocks(self, module):
        # Recursively inject advanced components matching the Custom TFT implementation
        for name, child in module.named_children():
            # Replace VSN
            if isinstance(child, VariableSelectionNetwork):
                if len(child.var_grns) > 0:
                    setattr(module, name, EnhancedVSN(child.joint_grn.lin_a.out_features, len(child.var_grns), child.var_grns[0].dropout.p))
                    # Recover exact Nixtla joint_grn dimensions safely
                    out_proj_size = child.joint_grn.out_proj.out_features if getattr(child.joint_grn, 'out_proj', None) is not None else None
                    lin_c_size = child.joint_grn.lin_c.in_features if getattr(child.joint_grn, 'lin_c', None) is not None else None
                    
                    getattr(module, name).vsn.joint_grn = EnhancedGRN(
                        input_size=child.joint_grn.lin_a.in_features,
                        hidden_size=child.joint_grn.lin_a.out_features,
                        output_size=out_proj_size,
                        context_hidden_size=lin_c_size,
                    )
            # Replace basic GRN
            elif isinstance(child, GRN):
                # Ensure we skip if it's already an EnhancedGRN. Joint GRN output size is sometimes different.
                out_size = child.out_proj.out_features if getattr(child, 'out_proj', None) is not None else None
                ctx_size = child.lin_c.in_features if getattr(child, 'lin_c', None) is not None else None
                setattr(module, name, EnhancedGRN(child.lin_a.in_features, child.lin_i.in_features, out_size, ctx_size, child.dropout.p))
            # Replace TFTEmbedding with Custom Continuous embedding
            elif isinstance(child, TFTEmbedding):
                new_embed = ContinuousTFTEmbedding(child.hidden_size, child.stat_input_size, child.futr_input_size, child.hist_input_size, child.tgt_size)
                if hasattr(child, 'tgt_embedding_vectors') and hasattr(new_embed.tft_embed, 'tgt_embedding_vectors'):
                    new_embed.tft_embed.tgt_embedding_vectors.data.copy_(child.tgt_embedding_vectors.data)
                if hasattr(child, 'tgt_embedding_bias') and hasattr(new_embed.tft_embed, 'tgt_embedding_bias'):
                    new_embed.tft_embed.tgt_embedding_bias.data.copy_(child.tgt_embedding_bias.data)
                setattr(module, name, new_embed)
            # Replace Interpretable Attention with Dual Attention Fusion
            elif isinstance(child, InterpretableMultiHeadAttention):
                # Deriving sequence_length safely instead of relying on private `child._mask`
                seq_len = self.seq_len + self.pred_len
                setattr(module, name, DualInterpretableMultiHeadAttention(child.n_head, child.qkv_linears.in_features, seq_len, child.attn_dropout.p, child.out_dropout.p))
            else:
                self._inject_enhanced_blocks(child)
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs):
        # Explicit ValueError checking for provided covariates matching the expected Nixtla configured size
        if x_mark_enc.shape[-1] != self.time_features_dim:
            raise ValueError(f"Provided encoder temporal features size {x_mark_enc.shape[-1]} != expected configuration {self.time_features_dim}")
        
        if x_mark_dec.shape[-1] != self.time_features_dim:
            raise ValueError(f"Provided decoder temporal features size {x_mark_dec.shape[-1]} != expected configuration {self.time_features_dim}")

        B, T, C = x_enc.shape
        insample_y = x_enc.permute(0, 2, 1).reshape(B * C, T)
        future_marks = x_mark_dec[:, -self.pred_len:, :]
        time_feats = torch.cat([x_mark_enc, future_marks], dim=1)
        futr_exog = time_feats.repeat_interleave(C, dim=0)

        windows_batch = {
            "insample_y": insample_y, 
            "futr_exog": futr_exog,   
            "hist_exog": None,        
            "stat_exog": None         
        }
        y_hat = self.nixtla_tft(windows_batch)
        y_hat = y_hat.squeeze(-1).view(B, C, self.pred_len).permute(0, 2, 1)
        
        # Post-Mixer cross-channel restoration
        if hasattr(self, 'cross_channel_mixer'):
            y_hat = self.cross_channel_mixer(y_hat)
            
        return y_hat

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, **kwargs):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs)
            return dec_out[:, -self.pred_len:, :]
        else:
            raise NotImplementedError(f"Nixtla TFT wrapper only supports forecasting tasks. Got {self.task_name}.")
