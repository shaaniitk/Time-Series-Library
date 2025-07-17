
from .base import BaseDecoder
from ..layers.enhanced_layers import EnhancedDecoderLayer
from models.EnhancedAutoformer_Fixed import EnhancedDecoder as StableAutoformerDecoder
import torch.nn as nn

class StableDecoder(BaseDecoder):
    """
    The stable Autoformer decoder, built with modular enhanced layers.
    """
    def __init__(self, d_layers, d_model, c_out, n_heads, d_ff, dropout, activation,
                 self_attention_comp, cross_attention_comp, decomp_comp, 
                 norm_layer=None, projection=None):
        super(StableDecoder, self).__init__()
        
        self.decoder = StableAutoformerDecoder(
            [
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
            ],
            norm_layer=norm_layer,
            projection=projection
        )

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        return self.decoder(x, cross, x_mask, cross_mask, trend)
