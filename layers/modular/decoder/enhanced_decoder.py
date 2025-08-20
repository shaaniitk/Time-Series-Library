
from .base import BaseDecoder
from .core_decoders import CoreEnhancedDecoder
from .decoder_output import DecoderOutput
from ..layers.enhanced_layers import EnhancedDecoderLayer
import torch.nn as nn

class EnhancedDecoder(BaseDecoder):
    """
    The enhanced Autoformer decoder, now built with modular layers.
    """
    def __init__(self, d_layers, d_model, c_out, n_heads, d_ff, dropout, activation,
                 self_attention_comp, cross_attention_comp, decomp_comp, 
                 norm_layer=None, projection=None):
        super(EnhancedDecoder, self).__init__()
        
        layers = [
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
        ]
        
        self.decoder = CoreEnhancedDecoder(
            layers=layers,
            d_model=d_model,
            norm_layer=norm_layer
        )

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        output = self.decoder(x, cross, x_mask, cross_mask, trend)
        return output.seasonal, output.trend
