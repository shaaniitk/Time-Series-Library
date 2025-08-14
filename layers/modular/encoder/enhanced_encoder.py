
from .base import BaseEncoder
from ..layers.enhanced_layers import EnhancedEncoderLayer
from layers.Autoformer_EncDec import Encoder as EnhancedAutoformerEncoder
import torch.nn as nn

class EnhancedEncoder(BaseEncoder):
    """
    The enhanced Autoformer encoder, now built with modular layers.
    """
    def __init__(self, e_layers, d_model, n_heads, d_ff, dropout, activation, 
                 attention_comp, decomp_comp, conv_layers=None, norm_layer=None):
        super(EnhancedEncoder, self).__init__()
        
        self.encoder = EnhancedAutoformerEncoder(
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
            conv_layers=conv_layers,
            norm_layer=norm_layer
        )

    def forward(self, x, attn_mask=None):
        return self.encoder(x, attn_mask)
