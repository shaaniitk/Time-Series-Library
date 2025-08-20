
from .base import BaseDecoder
from .core_decoders import CoreAutoformerDecoder
from .decoder_output import DecoderOutput
from ..layers.standard_layers import StandardDecoderLayer
import torch.nn as nn

class StandardDecoder(BaseDecoder):
    """
    The standard Autoformer decoder, now built with modular layers.
    """
    def __init__(self, d_layers, d_model, c_out, n_heads, d_ff, dropout, activation,
                 self_attention_comp, cross_attention_comp, decomp_comp, 
                 norm_layer=None, projection=None):
        super(StandardDecoder, self).__init__()
        
        layers = [
            StandardDecoderLayer(
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
        
        self.decoder = CoreAutoformerDecoder(
            layers=layers,
            norm_layer=norm_layer,
            projection=projection
        )

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        output = self.decoder(x, cross, x_mask, cross_mask, trend)
        return output.seasonal, output.trend
