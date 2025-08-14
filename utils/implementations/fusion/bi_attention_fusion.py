# FILE: utils/implementations/fusion/bi_attention_fusion.py
import torch
import torch.nn as nn
from layers.SelfAttention_Family import AttentionLayer, FullAttention

class BiDirectionalFusionProcessor(nn.Module):
    """
    Fuses target and covariate representations using bi-directional cross-attention.
    """
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        # Attention where targets query covariates
        self.target_to_covariate_attn = AttentionLayer(
            FullAttention(mask_flag=False, attention_dropout=dropout),
            d_model, n_heads)
        
        # Attention where covariates query targets
        self.covariate_to_target_attn = AttentionLayer(
            FullAttention(mask_flag=False, attention_dropout=dropout),
            d_model, n_heads)
        
        # MLP to process the final fused representation
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, target_repr, covariate_repr):
        # Let targets learn from covariates
        target_enriched, _ = self.target_to_covariate_attn(
            queries=target_repr,
            keys=covariate_repr,
            values=covariate_repr
        )
        
        # Let covariates learn from targets
        covariate_enriched, _ = self.covariate_to_target_attn(
            queries=covariate_repr,
            keys=target_repr,
            values=target_repr
        )
        
        # Add & Norm for both streams before final fusion
        target_out = self.norm1(target_repr + target_enriched)
        covariate_out = self.norm2(covariate_repr + covariate_enriched)
        
        # Combine the two enriched representations
        fused_repr = torch.cat([target_out, covariate_out], dim=-1)
        
        # Final processing MLP
        enriched_context = self.fusion_mlp(fused_repr)
        
        return enriched_context
