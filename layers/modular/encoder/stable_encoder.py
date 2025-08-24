from .abstract_encoder import BaseEncoder
from .base_encoder import ModularEncoder
from ..layers.enhanced_layers import EnhancedEncoderLayer
import torch.nn as nn

class StableEncoder(BaseEncoder):
    """
    The stable Autoformer encoder, built with modular enhanced layers.

    This encoder uses the stable Autoformer architecture, with a stack of
    EnhancedEncoderLayer instances. It is a wrapper around the ModularEncoder
    class, which provides the basic encoder structure.

    Args:
        num_encoder_layers (int): The number of encoder layers.
        d_model (int): The dimension of the model.
        n_heads (int): The number of attention heads.
        d_ff (int): The dimension of the feed-forward network.
        dropout (float): The dropout rate.
        activation (str): The activation function.
        attention_comp (nn.Module): The attention component.
        decomp_comp (nn.Module): The decomposition component.
        norm_layer (nn.Module, optional): The normalization layer. Defaults to None.
    """
    def __init__(self, num_encoder_layers=None, d_model=None, n_heads=None, d_ff=None, dropout=0.1, activation="gelu", 
                 attention_comp=None, decomp_comp=None, norm_layer=None, **kwargs):
        super(StableEncoder, self).__init__()
        if num_encoder_layers is None:
            num_encoder_layers = kwargs.pop('e_layers', 1)
        d_model = d_model if d_model is not None else kwargs.get('hidden_size', 32)
        n_heads = n_heads if n_heads is not None else kwargs.get('num_heads', 2)
        d_ff = d_ff if d_ff is not None else max(4 * int(d_model), 32)
        attention_comp = attention_comp if attention_comp is not None else kwargs.get('attention')
        decomp_comp = decomp_comp if decomp_comp is not None else kwargs.get('decomposition')

        self.encoder = ModularEncoder(
            [
                EnhancedEncoderLayer(
                    attention_comp,
                    decomp_comp,
                    d_model,
                    n_heads,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(int(num_encoder_layers))
            ],
            norm_layer=norm_layer
        )

    def forward(self, x, attn_mask=None):
        return self.encoder(x, attn_mask)
