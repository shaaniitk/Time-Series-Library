import torch
import torch.nn as nn
from .base import BaseEncoder
from models.HierarchicalEnhancedAutoformer import HierarchicalEncoder as HierarchicalAutoformerEncoder
from models.EnhancedAutoformer import EnhancedEncoder, EnhancedEncoderLayer
from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelationLayer, AdaptiveAutoCorrelation
from layers.MultiWaveletCorrelation import MultiWaveletTransform
from utils.logger import logger

class HierarchicalEncoder(BaseEncoder):
    """
    The hierarchical Autoformer encoder, now properly modularized.
    """
    def __init__(self, e_layers, d_model, n_heads, d_ff, dropout, activation, 
                 attention_type, decomp_type, decomp_params, n_levels=3, share_weights=False, **kwargs):
        super(HierarchicalEncoder, self).__init__()
        
        from argparse import Namespace
        from ..attention import get_attention_component

        mock_configs = Namespace(
            e_layers=e_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=dropout, activation=activation, factor=1
        )

        def attention_factory():
            return get_attention_component(attention_type, d_model=d_model, n_heads=n_heads, dropout=dropout)

        self.encoder = HierarchicalAutoformerEncoder(
            configs=mock_configs, 
            n_levels=n_levels, 
            share_weights=share_weights,
            _attention_factory=attention_factory,
            decomp_params=decomp_params
        )

    def forward(self, x, attn_mask=None):
        # The original hierarchical encoder expects a list of tensors
        if not isinstance(x, list):
            x = [x]
        multi_res_output = self.encoder(x, attn_mask)
        
        # For compatibility with standard decoder interface, 
        # we need to return a single tensor for cross-attention
        # We'll concatenate or take the finest resolution
        if isinstance(multi_res_output, list) and len(multi_res_output) > 0:
            # Use the finest resolution (last one) as the main output
            output = multi_res_output[-1]
        else:
            output = multi_res_output
            
        # Return output and None for attention weights to match standard encoder interface
        return output, None