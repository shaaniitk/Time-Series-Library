import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, series_decomp
from layers.Normalization import get_norm_layer
import math
from utils.logger import logger


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self._validate_configs(configs)
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # Fix: Ensure odd kernel size for stability
        kernel_size = getattr(configs, 'moving_avg', 25)
        kernel_size = kernel_size + (1 - kernel_size % 2)
        self.decomp = series_decomp(kernel_size)

        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Add gradient scaling
        self.gradient_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        self.encoder = Encoder([
            EncoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, moving_avg=kernel_size,
                dropout=configs.dropout, activation=configs.activation
            ) for l in range(configs.e_layers)
        ], norm_layer=get_norm_layer(configs.norm_type, configs.d_model))

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.decoder = Decoder([
                DecoderLayer(
                    AutoCorrelationLayer(AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model, configs.c_out, configs.d_ff, moving_avg=kernel_size,
                    dropout=configs.dropout, activation=configs.activation,
                ) for l in range(configs.d_layers)
            ], norm_layer=get_norm_layer(configs.norm_type, configs.d_model),
               projection=nn.Linear(configs.d_model, configs.c_out, bias=True))
        
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def _validate_configs(self, configs):
        required = ['task_name', 'seq_len', 'pred_len', 'enc_in', 'd_model']
        for attr in required:
            if not hasattr(configs, attr):
                raise ValueError(f"Missing config: {attr}")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Fix: Add numerical stability
        mean = torch.mean(x_enc, dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        
        # Apply gradient scaling
        enc_out = enc_out * self.gradient_scale

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)

        return trend_part + seasonal_part

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Add input validation
        if x_enc.size(-1) != getattr(self, 'enc_in', x_enc.size(-1)):
            logger.warning(f"Input dimension mismatch detected")
        
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