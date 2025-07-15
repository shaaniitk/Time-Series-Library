import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from layers.Attention import get_attention_layer
from layers.Normalization import get_norm_layer
import math


class AdaptiveDecomposition(nn.Module):
    """Multi-scale adaptive decomposition with frequency domain analysis"""
    
    def __init__(self, input_dim, scales=[3, 7, 15, 31]):
        super().__init__()
        self.scales = scales
        self.freq_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        
        # Learnable trend extractors for each scale
        self.trend_extractors = nn.ModuleList([
            nn.Conv1d(input_dim, input_dim, kernel_size=k, padding=k//2, groups=input_dim)
            for k in scales
        ])
        
        # Frequency domain filter
        self.freq_filter = nn.Parameter(torch.ones(1, 1, 1))
        
    def forward(self, x):
        B, L, D = x.shape
        x_freq = torch.fft.rfft(x, dim=1)
        
        # Apply frequency domain filtering
        x_freq = x_freq * self.freq_filter
        x_filtered = torch.fft.irfft(x_freq, n=L, dim=1)
        
        # Multi-scale trend extraction
        trends = []
        weights = F.softmax(self.freq_weights, dim=0)
        
        for i, extractor in enumerate(self.trend_extractors):
            trend = extractor(x_filtered.transpose(1, 2)).transpose(1, 2)
            trends.append(trend * weights[i])
        
        trend = sum(trends)
        seasonal = x - trend
        
        return seasonal, trend


class DynamicAttention(nn.Module):
    """Dynamic attention with adaptive receptive field"""
    
    def __init__(self, d_model, n_heads, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Dynamic position encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        # Adaptive attention weights
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        
        # Dynamic positional encoding
        pos = torch.arange(L, device=x.device).float().unsqueeze(-1) / L
        pos_emb = self.pos_encoder(pos).unsqueeze(0).expand(B, -1, -1)
        x = x + pos_emb
        
        # Multi-head attention with learnable temperature
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        attn = (q @ k.transpose(-2, -1)) * (self.temperature / math.sqrt(self.head_dim))
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1e9)
            
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        
        return self.out_proj(out), attn


class ResidualGate(nn.Module):
    """Gated residual connection with learnable skip weights"""
    
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x, residual):
        combined = torch.cat([x, residual], dim=-1)
        gate_weights = self.gate(combined)
        return gate_weights * x + (1 - gate_weights) * residual


class UltraEncoderLayer(nn.Module):
    """Ultra-enhanced encoder with multiple improvements"""
    
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = DynamicAttention(d_model, n_heads)
        self.decomp = AdaptiveDecomposition(d_model)
        self.gate = ResidualGate(d_model)
        
        # Enhanced FFN with GLU
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        # Self-attention with residual gating
        residual = x
        x = self.norm1(x)
        attn_out, attn = self.attention(x, attn_mask)
        x = self.gate(self.dropout(attn_out), residual)
        
        # Adaptive decomposition
        x, _ = self.decomp(x)
        
        # Enhanced FFN
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = self.gate(self.dropout(ffn_out), residual)
        
        return x, attn


class UltraAutoformer(nn.Module):
    """Ultra-enhanced Autoformer with advanced features"""
    
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        
        # Multi-scale decomposition
        self.decomp = AdaptiveDecomposition(configs.dec_in if hasattr(configs, 'dec_in') else configs.enc_in)
        
        # Enhanced embeddings with learnable scaling
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.embed_scale = nn.Parameter(torch.ones(1))
        
        # Ultra encoder layers
        self.encoder_layers = nn.ModuleList([
            UltraEncoderLayer(configs.d_model, configs.n_heads, configs.d_ff, configs.dropout)
            for _ in range(configs.e_layers)
        ])
        
        self.encoder_norm = get_norm_layer(configs.norm_type, configs.d_model)
        
        # Task-specific heads
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            
            # Ultra decoder with cross-attention
            self.decoder_layers = nn.ModuleList([
                UltraEncoderLayer(configs.d_model, configs.n_heads, configs.d_ff, configs.dropout)
                for _ in range(configs.d_layers)
            ])
            
            # Multi-head prediction
            self.prediction_heads = nn.ModuleList([
                nn.Linear(configs.d_model, configs.c_out)
                for _ in range(3)  # Ensemble of 3 heads
            ])
            
            self.head_weights = nn.Parameter(torch.ones(3) / 3)
            
        # Adaptive learning rate scaling
        self.lr_scale = nn.Parameter(torch.ones(1))
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Multi-scale decomposition
        seasonal_init, trend_init = self.decomp(x_dec)
        
        # Enhanced encoding
        enc_out = self.enc_embedding(x_enc, x_mark_enc) * self.embed_scale
        
        for layer in self.encoder_layers:
            enc_out, _ = layer(enc_out)
        
        enc_out = self.encoder_norm(enc_out)
        
        # Enhanced decoding
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        
        for layer in self.decoder_layers:
            dec_out, _ = layer(dec_out)
        
        # Multi-head ensemble prediction
        predictions = []
        weights = F.softmax(self.head_weights, dim=0)
        
        for i, head in enumerate(self.prediction_heads):
            pred = head(dec_out)
            predictions.append(pred * weights[i])
        
        seasonal_part = sum(predictions)
        
        # Add trend component
        trend_part = trend_init[:, -self.pred_len:, :] if trend_init.size(1) >= self.pred_len else trend_init
        
        return seasonal_part + trend_part
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        
        # Other tasks use standard encoding
        enc_out = self.enc_embedding(x_enc, x_mark_enc) * self.embed_scale
        
        for layer in self.encoder_layers:
            enc_out, _ = layer(enc_out)
        
        enc_out = self.encoder_norm(enc_out)
        
        if self.task_name == 'classification':
            # Global average pooling + classification head
            pooled = enc_out.mean(dim=1)
            return F.linear(pooled, self.prediction_heads[0].weight[:configs.num_class])
        
        return enc_out


Model = UltraAutoformer