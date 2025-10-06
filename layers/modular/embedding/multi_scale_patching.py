import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from layers.modular.embedding.patching import PatchingLayer

class MultiScalePatchingComposer(nn.Module):
    """
    Composes multiple patching layers to process a time series at different scales (patch lengths).
    The outputs from different scales are fused using a cross-attention mechanism, which handles
    varying numbers of patches from each scale without information loss.
    """
    def __init__(self, patch_configs: List[Dict[str, int]], d_model: int, input_features: int, num_latents: int = 64, n_heads: int = 8):
        """
        Args:
            patch_configs (List[Dict[str, int]]): A list of dictionaries, each specifying 
                                                  'patch_len' and 'stride' for a PatchingLayer.
            d_model (int): The dimensionality of the model.
            input_features (int): The number of input features in the time series.
            num_latents (int): The number of latent queries to use for cross-attention. This determines the output sequence length.
            n_heads (int): The number of heads for the multi-head attention.
        """
        super().__init__()
        self.d_model = d_model
        self.patch_configs = patch_configs
        self.num_latents = num_latents

        # Create a list of PatchingLayer instances
        self.patchers = nn.ModuleList([
            PatchingLayer(
                patch_len=config['patch_len'],
                stride=config['stride'],
                d_model=d_model,
                input_features=input_features
            ) for config in patch_configs
        ])

        # Learnable latent queries for cross-attention
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, d_model))

        # Cross-attention layer for each scale
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
            for _ in patch_configs
        ])

        # Self-attention layer to fuse the results from different scales
        self.fusion_self_attention = nn.MultiheadAttention(embed_dim=d_model * len(patch_configs), num_heads=n_heads, batch_first=True)
        
        # Final projection and normalization
        self.final_projection = nn.Linear(d_model * len(patch_configs), d_model)
        self.fusion_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply multi-scale patching and fuse the results using cross-attention.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, features].

        Returns:
            torch.Tensor: The fused patched output of shape [batch_size, num_latents, d_model].
            Dict[str, torch.Tensor]: A dictionary containing the raw outputs from each scale.
        """
        batch_size = x.shape[0]
        
        # Get patched outputs from each scale
        patched_outputs = [patcher(x) for patcher in self.patchers]

        # Apply cross-attention for each scale
        attended_outputs = []
        for i, patch_seq in enumerate(patched_outputs):
            # The latent queries attend to the patch sequence from this scale
            attended_output, _ = self.cross_attention_layers[i](
                query=self.latent_queries.expand(batch_size, -1, -1),
                key=patch_seq,
                value=patch_seq
            )
            attended_outputs.append(attended_output)

        # Concatenate the outputs from all scales along the feature dimension
        concatenated_attended = torch.cat(attended_outputs, dim=-1) # [B, num_latents, d_model * num_scales]

        # Fuse the concatenated outputs using self-attention
        fused_output, _ = self.fusion_self_attention(
            query=concatenated_attended,
            key=concatenated_attended,
            value=concatenated_attended
        )

        # Final projection to the model dimension
        projected_output = self.final_projection(fused_output)
        
        # Add residual connection and layer norm
        final_output = self.fusion_layer_norm(projected_output + attended_outputs[0]) # Residual from first scale

        # Store raw outputs for analysis
        raw_outputs_dict = {
            f"scale_{i}": out for i, out in enumerate(patched_outputs)
        }
        
        return final_output, raw_outputs_dict

    def get_config_info(self) -> Dict:
        """Returns information about the component's configuration."""
        return {
            "patch_configs": self.patch_configs,
            "d_model": self.d_model,
            "num_scales": len(self.patchers),
            "num_latents": self.num_latents
        }