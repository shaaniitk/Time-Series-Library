
from .base import BaseEncoder
from ..layers.enhanced_layers import EnhancedEncoderLayer
from models.EnhancedAutoformer_Fixed import EnhancedEncoder as StableAutoformerEncoder
import torch.nn as nn

class StableEncoder(BaseEncoder):
    """
    The stable Autoformer encoder, built with modular enhanced layers.
    """
    def __init__(self, e_layers, d_model, n_heads, d_ff, dropout, activation, 
                 attention_comp, decomp_comp, conv_layers=None, norm_layer=None):
        super(StableEncoder, self).__init__()
        
        self.encoder = StableAutoformerEncoder(
            [
                EnhancedEncoderLayer(
                    attention_comp,
                    decomp_comp,
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=norm_layer
        )

    def forward(self, x, attn_mask=None):
        return self.encoder(x, attn_mask)
