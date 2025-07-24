"""
ConvolutionalAttention: Convolutional attention mechanism combining spatial and temporal convolutions.
"""
import logging
import torch
import torch.nn as nn
from ..base import BaseConvolutionAdapter

logger = logging.getLogger(__name__)

class ConvolutionalAttention(BaseConvolutionAdapter):
    """
    Convolutional attention mechanism combining spatial and temporal convolutions.
    Uses 2D convolutions to capture both spatial (feature) and temporal relationships in attention computation.
    """
    def __init__(self, d_model, n_heads, conv_kernel_size=3, pool_size=2, dropout=0.1):
        super().__init__()
        logger.info(f"Initializing ConvolutionalAttention: d_model={d_model}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.conv_kernel_size = conv_kernel_size
        self.spatial_conv = nn.Conv2d(
            1, n_heads,
            kernel_size=(conv_kernel_size, conv_kernel_size),
            padding=(conv_kernel_size//2, conv_kernel_size//2)
        )
        self.temporal_conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.temporal_conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.temporal_conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def get_output_dim(self) -> int:
        return self.d_model

    def get_attention_type(self) -> str:
        return "convolutional"

    def apply_attention(self, queries, keys, values, attn_mask=None):
        return self.forward(queries, keys, values, attn_mask)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        residual = queries
        q_temp = self.temporal_conv_q(queries.transpose(1, 2)).transpose(1, 2)
        # ... (rest of forward logic as in original)
