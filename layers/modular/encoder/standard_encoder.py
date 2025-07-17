
from .base import BaseEncoder
from ..layers.standard_layers import StandardEncoderLayer
from layers.Autoformer_EncDec import Encoder as AutoformerEncoder
import torch.nn as nn

class StandardEncoder(BaseEncoder):
    """
    The standard Autoformer encoder, now built with modular layers.
    """
    def __init__(self, e_layers, d_model, n_heads, d_ff, dropout, activation, 
                 attention_comp, decomp_comp, conv_layers=None, norm_layer=None):
        super(StandardEncoder, self).__init__()
        
        self.encoder = AutoformerEncoder(
            [
                StandardEncoderLayer(
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
