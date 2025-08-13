"""
Flexible Autoformer using existing modular components
"""

import torch
import torch.nn as nn
from typing import List
from configs.modular_components import ModularAssembler, component_registry, register_all_components
from configs.schemas import ModularAutoformerConfig, ComponentType, AttentionConfig, DecompositionConfig

class FlexibleModularAutoformer(nn.Module):
    """
    Flexible Autoformer that routes different time series through different processing paths
    using existing modular components
    """
    
    def __init__(self, target_indices: List[int], covariate_groups: List[List[int]], config):
        super().__init__()
        self.target_indices = target_indices
        self.covariate_groups = covariate_groups  # List of groups, each group is list of indices
        self.config = config
        
        # Register components
        register_all_components()
        self.assembler = ModularAssembler(component_registry)
        
        self._init_processing_paths()
    
    def _init_processing_paths(self):
        """Initialize different processing paths using modular components"""
        
        # Wavelet decomposition for targets
        self.target_wavelet = component_registry.create_component(
            ComponentType.WAVELET_DECOMP,
            DecompositionConfig(
                type=ComponentType.WAVELET_DECOMP,
                wavelet_type='db4',
                levels=3
            ),
            d_model=len(self.target_indices)
        )
        
        # Enhanced attention for wavelet-processed targets
        self.target_attention = component_registry.create_component(
            ComponentType.ENHANCED_AUTOCORRELATION,
            AttentionConfig(
                type=ComponentType.ENHANCED_AUTOCORRELATION,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout
            )
        )
        
        # Self attention for each covariate group
        self.group_attentions = nn.ModuleList([
            component_registry.create_component(
                ComponentType.AUTOCORRELATION,
                AttentionConfig(
                    type=ComponentType.AUTOCORRELATION,
                    d_model=self.config.d_model,
                    n_heads=self.config.n_heads,
                    dropout=self.config.dropout
                )
            ) for _ in self.covariate_groups
        ])
        
        # Cross attention across covariate groups
        self.inter_group_attention = component_registry.create_component(
            ComponentType.CROSS_RESOLUTION,
            AttentionConfig(
                type=ComponentType.CROSS_RESOLUTION,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout
            )
        )
        
        # Cross attention between covariates and targets
        self.covariate_target_attention = component_registry.create_component(
            ComponentType.CROSS_RESOLUTION,
            AttentionConfig(
                type=ComponentType.CROSS_RESOLUTION,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout
            )
        )
        
        # Projections
        self.target_proj = nn.Linear(len(self.target_indices), self.config.d_model)
        self.group_projs = nn.ModuleList([
            nn.Linear(len(group), self.config.d_model) for group in self.covariate_groups
        ])
        self.output_proj = nn.Linear(self.config.d_model, len(self.target_indices))
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass with hierarchical covariate group processing"""
        
        # Process targets: wavelet → self attention
        targets = x_enc[:, :, self.target_indices]
        target_seasonal, target_trend = self.target_wavelet(targets)
        target_processed = target_seasonal + target_trend
        target_embedded = self.target_proj(target_processed)
        target_attended, _ = self.target_attention(
            target_embedded, target_embedded, target_embedded, mask
        )
        
        # Process each covariate group with self attention
        group_outputs = []
        for i, (group_indices, group_attention, group_proj) in enumerate(
            zip(self.covariate_groups, self.group_attentions, self.group_projs)
        ):
            group_data = x_enc[:, :, group_indices]  # [B, L, group_size]
            group_embedded = group_proj(group_data)  # [B, L, d_model]
            group_attended, _ = group_attention(
                group_embedded, group_embedded, group_embedded, mask
            )
            group_outputs.append(group_attended)
        
        # Cross attention across covariate groups
        if len(group_outputs) > 1:
            # Concatenate all groups for inter-group attention
            all_groups = torch.stack(group_outputs, dim=2)  # [B, L, n_groups, d_model]
            B, L, n_groups, d_model = all_groups.shape
            all_groups_flat = all_groups.view(B, L * n_groups, d_model)
            
            inter_group_attended, _ = self.inter_group_attention(
                all_groups_flat, all_groups_flat, all_groups_flat, mask
            )
            # Reshape back and average across groups
            inter_group_reshaped = inter_group_attended.view(B, L, n_groups, d_model)
            covariate_consolidated = inter_group_reshaped.mean(dim=2)  # [B, L, d_model]
        else:
            covariate_consolidated = group_outputs[0]
        
        # Cross attention between consolidated covariates and targets
        final_output, _ = self.covariate_target_attention(
            target_attended, covariate_consolidated, covariate_consolidated, mask
        )
        
        # Project to target dimensions
        output = self.output_proj(final_output)
        return output[:, -self.config.pred_len:, :]

# Factory function
def create_flexible_modular_autoformer(
    target_indices: List[int],
    covariate_groups: List[List[int]],  # List of groups, each group is list of indices
    seq_len: int = 96,
    pred_len: int = 24,
    **kwargs
):
    """Create flexible autoformer with hierarchical covariate group processing"""
    from argparse import Namespace
    
    config = Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=kwargs.get('d_model', 512),
        n_heads=kwargs.get('n_heads', 8),
        dropout=kwargs.get('dropout', 0.1)
    )
    
    return FlexibleModularAutoformer(target_indices, covariate_groups, config)

# Usage
if __name__ == "__main__":
    try:
        print("Testing FlexibleModularAutoformer...")
        
        # Simplified test: 2 groups of 2 covariates each
        covariate_groups = [[0, 1], [2, 3]]
        
        model = create_flexible_modular_autoformer(
            target_indices=[4, 5],
            covariate_groups=covariate_groups,
            seq_len=96,
            pred_len=24,
            d_model=64
        )
        
        print("Model created successfully")
        
        # Test
        x_enc = torch.randn(2, 96, 6)  # 4 covariates + 2 targets
        x_mark_enc = torch.randn(2, 96, 4)
        x_dec = torch.randn(2, 48, 6)
        x_mark_dec = torch.randn(2, 48, 4)
        
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f"Output shape: {output.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()