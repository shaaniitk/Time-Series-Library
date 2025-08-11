import torch
import torch.nn as nn
from .base import BaseEncoder
from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelationLayer, AdaptiveAutoCorrelation
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

        # Build per-level encoder stack manually (lightweight) to avoid depending on enhancedcomponents version
        self.levels = n_levels
        self.share_weights = share_weights
        self.decomp_params = decomp_params
        self.attention_type = attention_type
        self.attention_kwargs = dict(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.activation = activation
        self.d_model = d_model
        self.d_ff = d_ff

        attention_comp = get_attention_component(attention_type, **self.attention_kwargs)
        from ..decomposition import get_decomposition_component
        decomp_comp = get_decomposition_component(decomp_type, **decomp_params)

        from ..layers.enhanced_layers import EnhancedEncoderLayer
        layer = EnhancedEncoderLayer(attention_comp, decomp_comp, d_model, d_ff, dropout=dropout, activation=activation)
        self.layer = layer  # base layer
        self.encoders = nn.ModuleList([layer] if share_weights else [
            EnhancedEncoderLayer(
                get_attention_component(attention_type, **self.attention_kwargs),
                get_decomposition_component(decomp_type, **decomp_params),
                d_model,
                d_ff,
                dropout=dropout,
                activation=activation
            ) for _ in range(n_levels)
        ])

    def forward(self, x, attn_mask=None):
        # The original hierarchical encoder expects a list of tensors
        if not isinstance(x, list):
            x = [x]
        # Simple placeholder multi-resolution processing: progressively average pool
        outputs = []
        current = x[0]
        for enc in self.encoders:
            enc_out, _ = enc(current, attn_mask=attn_mask)
            outputs.append(enc_out)
            if enc_out.size(1) > 2:
                current = torch.nn.functional.avg_pool1d(enc_out.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            else:
                current = enc_out
        # Return finest resolution output for interface compatibility
        finest = outputs[-1]
        # Upsample if sequence length shrank due to pooling to maintain interface compatibility
        if finest.size(1) != x[0].size(1):
            finest = torch.nn.functional.interpolate(finest.transpose(1, 2), size=x[0].size(1), mode="linear", align_corners=False).transpose(1, 2)
        return finest, None