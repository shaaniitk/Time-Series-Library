"""
Abstract base class for all Fourier-based attention mechanisms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ...base_interfaces import BaseAttention

class FourierAttentionBase(BaseAttention):
    """Abstract base for Fourier-based attention mechanisms."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)

    def get_output_dim(self) -> int:
        return self.d_model

    def get_attention_type(self) -> str:
        return "fourier"
