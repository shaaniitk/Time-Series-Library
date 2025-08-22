import torch
import torch.nn as nn
from .abstract_encoder import BaseEncoder
from ..layers.enhanced_layers import EnhancedEncoderLayer
from ..core import get_attention_component
from ..decomposition import get_decomposition_component
from utils.logger import logger

class HierarchicalEncoder(BaseEncoder):
    """
    The hierarchical Autoformer encoder, now properly modularized.
    """
    def __init__(self, num_encoder_layers, d_model, n_heads, d_ff, dropout, activation, 
                 attention_comp, decomp_comp, hierarchical_config, norm_layer=None):
        super(HierarchicalEncoder, self).__init__()
        
        self.levels = hierarchical_config.n_levels
        self.share_weights = hierarchical_config.level_configs is None

        self.encoders = nn.ModuleList()
        for i in range(self.levels):
            if self.share_weights:
                layer = EnhancedEncoderLayer(
                    attention_comp,
                    decomp_comp,
                    d_model,
                    n_heads,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
            else:
                # This part is not fully implemented as the level_configs is not used
                # For now, it will create independent layers for each level
                layer = EnhancedEncoderLayer(
                    get_attention_component(attention_comp.type, d_model=d_model, n_heads=n_heads, dropout=dropout),
                    get_decomposition_component(decomp_comp.type, kernel_size=decomp_comp.kernel_size),
                    d_model,
                    n_heads,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
            self.encoders.append(layer)

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
