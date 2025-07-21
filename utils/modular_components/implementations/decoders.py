"""
UNIFIED DECODER COMPONENTS
All decoder mechanisms in one place - clean modular structure
"""

import torch.nn as nn
from typing import Optional, Tuple

from .Layers import StandardDecoderLayer, EnhancedDecoderLayer
from layers.Autoformer_EncDec import Decoder as AutoformerDecoder
from utils.logger import logger
from abc import ABC, abstractmethod

class BaseDecoder(nn.Module, ABC):
    def __init__(self):
        super(BaseDecoder, self).__init__()

    @abstractmethod
    def forward(self, x: nn.Module, cross: nn.Module, x_mask: Optional[nn.Module] = None, cross_mask: Optional[nn.Module] = None, trend: Optional[nn.Module] = None) -> Tuple[nn.Module, nn.Module]:
        raise NotImplementedError

class StandardDecoder(BaseDecoder):
    def __init__(self, d_layers, d_model, c_out, n_heads, d_ff, dropout, activation, self_attention_comp, cross_attention_comp, decomp_comp, norm_layer=None, projection=None):
        super(StandardDecoder, self).__init__()
        self.decoder = AutoformerDecoder(
            [StandardDecoderLayer(self_attention_comp, cross_attention_comp, decomp_comp, d_model, c_out, d_ff, dropout=dropout, activation=activation) for _ in range(d_layers)],
            norm_layer=norm_layer,
            projection=projection
        )
    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        return self.decoder(x, cross, x_mask, cross_mask, trend)

class EnhancedDecoder(BaseDecoder):
    def __init__(self, d_layers, d_model, c_out, n_heads, d_ff, dropout, activation, self_attention_comp, cross_attention_comp, decomp_comp, norm_layer=None, projection=None):
        super(EnhancedDecoder, self).__init__()
        self.layers = nn.ModuleList([EnhancedDecoderLayer(self_attention_comp, cross_attention_comp, decomp_comp, d_model, c_out, d_ff, dropout=dropout, activation=activation) for _ in range(d_layers)])
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

class StableDecoder(BaseDecoder):
    def __init__(self, d_layers, d_model, c_out, n_heads, d_ff, dropout, activation, self_attention_comp, cross_attention_comp, decomp_comp, norm_layer=None, projection=None):
        super(StableDecoder, self).__init__()
        self.decoder = nn.ModuleList([EnhancedDecoderLayer(self_attention_comp, cross_attention_comp, decomp_comp, d_model, c_out, d_ff, dropout=dropout, activation=activation) for _ in range(d_layers)])
        self.norm = norm_layer
        self.projection = projection
    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.decoder:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x, trend

DECODER_REGISTRY = {
    "standard": StandardDecoder,
    "enhanced": EnhancedDecoder,
    "stable": StableDecoder,
}

def get_decoder_component(name: str, **kwargs):
    if name not in DECODER_REGISTRY:
        raise ValueError(f"Decoder '{name}' not found in registry.")
    return DECODER_REGISTRY[name](**kwargs)