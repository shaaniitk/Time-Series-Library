
from .base import BaseDecoder
from ..layers.enhanced_layers import EnhancedDecoderLayer

import torch.nn as nn

class StableDecoder(BaseDecoder):
    """
    The stable Autoformer decoder, built with modular enhanced layers.
    """
    def __init__(self, d_layers, d_model, c_out, n_heads, d_ff, dropout, activation,
                 self_attention_comp, cross_attention_comp, decomp_comp, 
                 norm_layer=None, projection=None):
        super(StableDecoder, self).__init__()
        
        self.layers = nn.ModuleList([
                EnhancedDecoderLayer(
                    self_attention_comp,
                    cross_attention_comp,
                    decomp_comp,
                    d_model,
                    c_out,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ])
        self.norm_layer = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        if trend is None:
            trend = torch.zeros_like(x)
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask, cross_mask)
            trend = trend + residual_trend
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.projection is not None:
            x = self.projection(x)
        return x + trend
