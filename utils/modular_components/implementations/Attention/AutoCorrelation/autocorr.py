"""
AutoCorrelation-based attention for time series (modularized).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseAutoCorrelationAttention

class AutoCorrelationAttention(BaseAutoCorrelationAttention):
    """Clean implementation of AutoCorrelation attention."""
    def __init__(self, d_model=512, n_heads=8, factor=1, dropout=0.1):
        class Config:
            def __init__(self):
                self.d_model = d_model
                self.n_heads = n_heads
                self.factor = factor
                self.dropout = dropout
        super().__init__(Config())
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def get_output_dim(self) -> int:
        return self.d_model

    def get_attention_type(self) -> str:
        return "autocorrelation"

    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)

    def time_delay_agg_training(self, values, corr):
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr)
        else:
            V = values.permute(0, 2, 3, 1).contiguous()
        V = V.permute(0, 3, 1, 2).contiguous()
        return V.contiguous(), corr
