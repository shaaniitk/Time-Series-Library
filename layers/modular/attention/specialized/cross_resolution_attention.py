import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from ..base import BaseAttention

class CrossResolutionAttention(BaseAttention):
    """
    Handles inputs from multiple resolutions. Its primary method,
    `forward_multi_resolution`, takes a list of feature tensors at different
    scales and fuses them. The standard `forward` method provides a basic
    compatibility layer.
    """
    def __init__(self, d_model: int, n_heads: int, num_levels: int = 3, dropout: float = 0.1, **kwargs):
        super().__init__(d_model, n_heads)
        self.num_levels = num_levels

        # Standard attention layers for fusing information between adjacent levels
        self.cross_res_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(num_levels - 1)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_levels - 1)])

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        A basic compatibility forward pass. For this component's main functionality,
        use the `forward_multi_resolution` method. This implementation performs
        standard cross-attention.
        """
        # This is a simple placeholder for API compatibility.
        attn_output, attn_weights = F.multi_head_attention_forward(
            query=queries, key=keys, value=values,
            embed_dim_to_check=self.d_model,
            num_heads=self.n_heads,
            in_proj_weight=torch.empty(0), in_proj_bias=torch.empty(0),
            bias_k=None, bias_v=None, add_zero_attn=False,
            dropout_p=0.0, out_proj_weight=self.w_o.weight, out_proj_bias=self.w_o.bias,
            training=self.training, key_padding_mask=None, need_weights=True,
            attn_mask=attn_mask
        )
        return attn_output, attn_weights

    def forward_multi_resolution(
        self,
        multi_res_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Applies cross-resolution attention between features from different scales.

        Args:
            multi_res_features (List[torch.Tensor]): A list of tensors, typically
                ordered from coarsest to finest resolution.
        
        Returns:
            torch.Tensor: The fused feature tensor at the finest resolution.
        """
        if len(multi_res_features) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} feature tensors, but got {len(multi_res_features)}")
        
        # Start with the coarsest resolution
        fused_features = multi_res_features[0]
        
        # Iteratively fuse finer resolutions
        for i in range(self.num_levels - 1):
            coarse_features = fused_features
            fine_features = multi_res_features[i + 1]
            
            # Upsample the coarser features to match the finer resolution
            if coarse_features.size(1) != fine_features.size(1):
                coarse_upsampled = F.interpolate(
                    coarse_features.transpose(1, 2),
                    size=fine_features.size(1),
                    mode='linear'
                ).transpose(1, 2)
            else:
                coarse_upsampled = coarse_features

            # Use the fine features as queries and the upsampled coarse features as keys/values
            attn_output, _ = self.cross_res_attention[i](
                query=fine_features,
                key=coarse_upsampled,
                value=coarse_upsampled
            )
            
            # Fuse with a residual connection
            fused_features = self.layer_norms[i](fine_features + attn_output)
            
        return fused_features