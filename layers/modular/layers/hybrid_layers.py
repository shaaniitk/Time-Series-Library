# FILE: layers/modular/layers/hybrid_layers.py
import torch
import torch.nn as nn

class HybridAttentionLayer(nn.Module):
    """
    A layer that combines two attention mechanisms in parallel.
    The outputs are concatenated and passed through a fusion layer.
    """
    def __init__(self, attention1, attention2, d_model, n_heads):
        super(HybridAttentionLayer, self).__init__()
        self.attention1 = attention1
        self.attention2 = attention2
        
        # Each attention layer outputs d_model. Concatenating them gives 2 * d_model.
        # This linear layer fuses them back to the original d_model dimension.
        self.fusion_layer = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, queries, keys, values, attn_mask=None):
        # Process through the first attention mechanism
        out1, attn1 = self.attention1(queries, keys, values, attn_mask=attn_mask)
        
        # Process through the second attention mechanism
        out2, attn2 = self.attention2(queries, keys, values, attn_mask=attn_mask)
        
        # Concatenate the outputs
        combined_output = torch.cat([out1, out2], dim=-1)
        
        # Fuse the combined output
        fused_output = self.fusion_layer(combined_output)
        
        # Add & Norm (standard transformer block structure)
        # Note: The original query is added to the fused output, a common residual connection
        output = self.norm(queries + fused_output)
        
        # Note: Returning a single set of attention weights is non-trivial.
        # For now, we return None. If visualization is needed, this can be adapted.
        return output, None
