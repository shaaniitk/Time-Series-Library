class BayesianLinear(nn.Module):
    """
    Bayesian Linear layer with weight uncertainty (ported from layers/BayesianLayers.py).
    Instead of fixed weights, this layer learns weight distributions and samples from them during forward passes to estimate uncertainty.
    """
    def __init__(self, in_features, out_features, bias=True, prior_std=1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.full((out_features, in_features), -5.0))
        if bias:
            self.bias_mean = nn.Parameter(torch.randn(out_features) * 0.1)
            self.bias_logvar = nn.Parameter(torch.full((out_features,), -5.0))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_logvar', None)

    def forward(self, x):
        # Sample weights and bias
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight = self.weight_mean + weight_std * torch.randn_like(self.weight_mean)
        if self.bias_mean is not None:
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mean + bias_std * torch.randn_like(self.bias_mean)
        else:
            bias = None
        return F.linear(x, weight, bias)
class MixtureOfExperts(nn.Module):
    """
    A generalized Mixture of Experts (MoE) component (ported from layers/MixtureOfExperts.py).
    This component can wrap any set of 'expert' modules and uses a gating
    network to dynamically select and combine their outputs.
    """
    def __init__(self, input_dim: int, num_experts: int, expert_class: type, expert_configs: list):
        super().__init__()
        assert num_experts == len(expert_configs), "Number of experts must match the number of expert configurations."
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        self.experts = nn.ModuleList([expert_class(**config) for config in expert_configs])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, ...]
        gate_scores = self.gating_network(x)
        expert_outputs = [expert(x) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=-1)  # [..., num_experts]
        output = (stacked * gate_scores.unsqueeze(-2)).sum(dim=-1)
        return output
"""
UNIFIED LAYER COMPONENTS
Complete layer implementations including normalization, residuals, and feed-forward.
This file is the single source of truth for all basic building block layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import logging

from ..base_interfaces import BaseComponent
from ..config_schemas import ComponentConfig
from .attention_clean import AutoCorrelationAttention, MultiHeadAttention
from .decomposition_unified import get_decomposition_component

logger = logging.getLogger(__name__)

# Abstract Base Classes for Layers
class BaseEncoderLayer(nn.Module):
    def __init__(self):
        super(BaseEncoderLayer, self).__init__()

    def forward(self, x: nn.Module, attn_mask: Optional[nn.Module] = None) -> Tuple[nn.Module, Optional[nn.Module]]:
        raise NotImplementedError

class BaseDecoderLayer(nn.Module):
    def __init__(self):
        super(BaseDecoderLayer, self).__init__()

    def forward(self, x: nn.Module, cross: nn.Module, x_mask: Optional[nn.Module] = None, cross_mask: Optional[nn.Module] = None) -> Tuple[nn.Module, nn.Module]:
        raise NotImplementedError

# Standard Autoformer Layers
class StandardEncoderLayer(BaseEncoderLayer):
    def __init__(self, attention_component, decomposition_component, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(StandardEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention_component
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = decomposition_component
        self.decomp2 = decomposition_component
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn

class StandardDecoderLayer(BaseDecoderLayer):
    def __init__(self, self_attention_comp, cross_attention_comp, decomposition_comp, d_model, c_out, d_ff=None, dropout=0.1, activation="relu"):
        super(StandardDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention_comp
        self.cross_attention = cross_attention_comp
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = decomposition_comp
        self.decomp2 = decomposition_comp
        self.decomp3 = decomposition_comp
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)
        residual_trend = trend1 + trend2 + trend3
        # Always return a tuple (x, residual_trend)
        if isinstance(x, tuple) and len(x) == 2:
            return x
        return x, residual_trend

# Enhanced Autoformer Layers
class EnhancedEncoderLayer(BaseEncoderLayer):
    def __init__(self, attention_component, decomposition_component, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EnhancedEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention_component
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.gate = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=1), nn.Sigmoid())
        self.decomp1 = decomposition_component
        self.decomp2 = decomposition_component
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation, F.relu)
        self.attention_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.ffn_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(self.attention_scale * new_x)
        x, _ = self.decomp1(x)
        residual = x
        y = x.transpose(-1, 1)
        gate_values = self.gate(y)
        y = y * gate_values
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y)).transpose(-1, 1)
        x = residual + self.ffn_scale * y
        res, _ = self.decomp2(x)
        return res, attn

class EnhancedDecoderLayer(BaseDecoderLayer):
    def __init__(self, self_attention_comp, cross_attention_comp, decomposition_comp, d_model, c_out, d_ff=None, dropout=0.1, activation="relu"):
        super(EnhancedDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention_comp
        self.cross_attention = cross_attention_comp
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = decomposition_comp
        self.decomp2 = decomposition_comp
        self.decomp3 = decomposition_comp
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=False)
        )
        self.activation = getattr(F, activation, F.relu)
        self.self_attn_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.cross_attn_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        residual = x
        new_x, _ = self.self_attention(x, x, x, attn_mask=x_mask)
        x = residual + self.dropout(self.self_attn_scale * new_x)
        x, trend1 = self.decomp1(x)
        residual = x
        new_x, _ = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = residual + self.dropout(self.cross_attn_scale * new_x)
        x, trend2 = self.decomp2(x)
        residual = x
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = residual + y
        x, trend3 = self.decomp3(x)
        residual_trend = trend1 + trend2 + trend3
        # Always return a tuple (x, residual_trend)
        if isinstance(x, tuple) and len(x) == 2:
            return x
        return x, residual_trend