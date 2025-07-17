
from .base import BaseDecoder
from ..layers.standard_layers import StandardDecoderLayer
from layers.Autoformer_EncDec import Decoder as AutoformerDecoder
import torch.nn as nn

class StandardDecoder(BaseDecoder):
    """
    The standard Autoformer decoder, now built with modular layers.
    """
    def __init__(self, d_layers, d_model, c_out, n_heads, d_ff, dropout, activation,
                 self_attention_comp, cross_attention_comp, decomp_comp, 
                 norm_layer=None, projection=None):
        super(StandardDecoder, self).__init__()
        
        self.decoder = AutoformerDecoder(
            [
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
            ],
            norm_layer=norm_layer,
            projection=projection
        )

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        return self.decoder(x, cross, x_mask, cross_mask, trend)
