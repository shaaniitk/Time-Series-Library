import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from layers.Attention import get_attention_layer
from layers.Normalization import get_norm_layer
from utils.logger import logger
from typing import Optional, List


class StableSeriesDecomp(nn.Module):
    def __init__(self, input_dim, kernel_size=25, eps=1e-8):
        super().__init__()
        self.kernel_size = kernel_size + (1 - kernel_size % 2)  # Ensure odd
        self.eps = eps
        self.trend_weights = nn.Parameter(torch.ones(self.kernel_size) / self.kernel_size)
        
    def forward(self, x):
        weights = F.softmax(self.trend_weights, dim=0)
        padding = self.kernel_size // 2
        
        # Efficient convolution - handle multiple channels
        B, L, D = x.shape
        x_padded = F.pad(x.transpose(1, 2), (padding, padding), mode='replicate')  # [B, D, L+padding]
        
        # Expand weights for all channels
        weights_expanded = weights.view(1, 1, -1).expand(D, 1, -1)  # [D, 1, kernel_size]
        trend = F.conv1d(x_padded, weights_expanded, groups=D, padding=0).transpose(1, 2)  # [B, L, D]
        seasonal = x - trend
        
        return seasonal, trend


class EnhancedEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        
        self.conv1 = nn.Conv1d(d_model, d_ff, 1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1, bias=False)
        self.decomp1 = StableSeriesDecomp(d_model)
        self.decomp2 = StableSeriesDecomp(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation, F.relu)
        self.attn_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(self.attn_scale * new_x)
        x, _ = self.decomp1(x)
        
        residual = x
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = residual + y
        res, _ = self.decomp2(x)
        
        return res, attn


class EnhancedEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class EnhancedDecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        self.conv1 = nn.Conv1d(d_model, d_ff, 1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1, bias=False)
        
        self.decomp1 = StableSeriesDecomp(d_model)
        self.decomp2 = StableSeriesDecomp(d_model)
        self.decomp3 = StableSeriesDecomp(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(d_model, c_out, 3, stride=1, padding=1, padding_mode='circular', bias=False)
        self.activation = getattr(F, activation, F.relu)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        
        residual = x
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(residual + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        
        return x, residual_trend


class EnhancedDecoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
            
        return x, trend


class EnhancedAutoformer(nn.Module):
    def __init__(self, configs, quantile_levels: Optional[List[float]] = None):
        super().__init__()
        self._validate_configs(configs)
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        # Fix: Use correct dimensions
        self.num_target_variables = getattr(configs, 'c_out_evaluation', configs.c_out)
        self.decomp = StableSeriesDecomp(configs.dec_in)

        # Quantile support
        self.is_quantile_mode = hasattr(configs, 'quantile_levels') and configs.quantile_levels
        self.num_quantiles = len(configs.quantile_levels) if self.is_quantile_mode else 1

        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Enhanced encoder
        enhanced_layers = []
        for _ in range(configs.e_layers):
            autocorr_layer = get_attention_layer(configs)
            enhanced_layer = EnhancedEncoderLayer(autocorr_layer, configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation)
            enhanced_layers.append(enhanced_layer)

        self.encoder = EnhancedEncoder(enhanced_layers, norm_layer=get_norm_layer(configs.norm_type, configs.d_model))

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            
            decoder_layers = []
            for _ in range(configs.d_layers):
                self_attn = get_attention_layer(configs)
                cross_attn = get_attention_layer(configs)
                decoder_layer = EnhancedDecoderLayer(self_attn, cross_attn, self.d_model, self.num_target_variables, configs.d_ff, dropout=configs.dropout, activation=configs.activation)
                decoder_layers.append(decoder_layer)

            self.decoder = EnhancedDecoder(decoder_layers, norm_layer=get_norm_layer(configs.norm_type, configs.d_model), projection=nn.Linear(self.d_model, configs.c_out, bias=True))

        # Other tasks
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def _validate_configs(self, configs):
        required = ['task_name', 'seq_len', 'pred_len', 'enc_in', 'dec_in', 'd_model', 'c_out']
        for attr in required:
            if not hasattr(configs, attr):
                raise ValueError(f"Missing config: {attr}")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Fix: Proper dimension handling
        seasonal_init_full, trend_init_full = self.decomp(x_dec)
        
        # Extract target features only for trend accumulation
        trend_init_targets = trend_init_full[:, :, :self.num_target_variables]
        mean_targets = torch.mean(x_dec[:, :, :self.num_target_variables], dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        
        trend_for_decoder = torch.cat([trend_init_targets[:, -self.label_len:, :], mean_targets], dim=1)
        
        # Full seasonal for embedding
        zeros_full = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_for_embedding = torch.cat([seasonal_init_full[:, -self.label_len:, :], zeros_full], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(seasonal_for_embedding, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_for_decoder)

        # Handle quantile mode
        if self.is_quantile_mode and self.num_quantiles > 1:
            trend_expanded = trend_part.unsqueeze(-1).repeat(1, 1, 1, self.num_quantiles).view(seasonal_part.shape)
            return trend_expanded + seasonal_part
        
        return trend_part + seasonal_part

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        
        if self.task_name == 'imputation':
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, _ = self.encoder(enc_out, attn_mask=None)
            return self.projection(enc_out)
        
        if self.task_name == 'anomaly_detection':
            enc_out = self.enc_embedding(x_enc, None)
            enc_out, _ = self.encoder(enc_out, attn_mask=None)
            return self.projection(enc_out)
        
        if self.task_name == 'classification':
            enc_out = self.enc_embedding(x_enc, None)
            enc_out, _ = self.encoder(enc_out, attn_mask=None)
            output = self.act(enc_out)
            output = self.dropout(output)
            output = output * x_mark_enc.unsqueeze(-1)
            output = output.reshape(output.shape[0], -1)
            return self.projection(output)
        
        return None


Model = EnhancedAutoformer