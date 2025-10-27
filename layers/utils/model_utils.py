"""
Model utilities for Enhanced SOTA PGAT
Contains configuration management, tensor utilities, and helper functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List


class ConfigManager:
    """Manages model configuration validation and setup"""
    
    @staticmethod
    def ensure_config_attributes(config):
        """Ensure config has all required attributes for parent class."""
        # CRITICAL FIX: Use actual training values instead of small defaults
        required_attrs = {
            'seq_len': getattr(config, 'seq_len', 256),  # Use training sequence length
            'pred_len': getattr(config, 'pred_len', 24), # Use training prediction length
            'enc_in': getattr(config, 'enc_in', 118),    # Use actual feature count
            'c_out': getattr(config, 'c_out', 4),        # Use actual target count
            'd_model': getattr(config, 'd_model', 128),   # Use training model dimension
            'n_heads': getattr(config, 'n_heads', 4),    # Use training head count
            'dropout': getattr(config, 'dropout', 0.1)
        }
        
        for attr, default_value in required_attrs.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                setattr(config, attr, default_value)
        
        # Enhanced PGAT specific attributes
        enhanced_attrs = {
            'num_wave_features': None,  # Will be inferred if not provided
            'use_multi_scale_patching': True,
            'use_hierarchical_mapper': True,
            'use_stochastic_learner': True,
            'use_gated_graph_combiner': True,
            'use_mixture_decoder': True
        }
        
        for attr, default_value in enhanced_attrs.items():
            if not hasattr(config, attr):
                setattr(config, attr, default_value)
    
    @staticmethod
    def validate_enhanced_config(config, num_wave_features: Optional[int] = None):
        """Validate enhanced model configuration parameters."""
        if getattr(config, 'use_multi_scale_patching', True):
            patch_configs = getattr(config, 'patch_configs', [])
            if not isinstance(patch_configs, list) or not all(isinstance(c, dict) for c in patch_configs):
                raise ValueError("patch_configs must be a list of dictionaries.")
            
            # Validate feature dimensions make sense
            if num_wave_features is not None:
                total_features = getattr(config, 'enc_in', 7)
                target_features = getattr(config, 'c_out', 3)
                if num_wave_features + target_features > total_features:
                    raise ValueError(
                        f"num_wave_features ({num_wave_features}) + c_out ({target_features}) "
                        f"cannot exceed enc_in ({total_features})"
                    )

        if getattr(config, 'use_hierarchical_mapper', True):
            n_heads = getattr(config, 'n_heads', 8)
            d_model = getattr(config, 'd_model', 512)
            if d_model % n_heads != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        if getattr(config, 'use_mixture_decoder', True):
            if not hasattr(config, 'c_out') or not hasattr(config, 'd_model'):
                raise ValueError("config must have c_out and d_model for MixtureDensityDecoder")


class TensorUtils:
    """Tensor manipulation utilities"""
    
    @staticmethod
    def align_sequence_lengths(wave_embedded: torch.Tensor, target_embedded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensure consistent sequence lengths for hierarchical mapping."""
        wave_seq_len = wave_embedded.shape[1]
        target_seq_len = target_embedded.shape[1]
        
        if wave_seq_len != target_seq_len:
            # Align to shorter sequence to avoid dimension mismatches
            min_len = min(wave_seq_len, target_seq_len)
            wave_embedded = wave_embedded[:, -min_len:, :]
            target_embedded = target_embedded[:, -min_len:, :]
            print(f"Warning: Aligned sequence lengths from {wave_seq_len}, {target_seq_len} to {min_len}")
        
        return wave_embedded, target_embedded


class ProjectionManager(nn.Module):
    """Manages dynamic projection layers to prevent memory issues"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Pre-allocate common projection sizes to avoid dynamic layer creation
        self.context_projection_layers = nn.ModuleDict()
        common_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
        for size in common_sizes:
            self.context_projection_layers[f'proj_{size}'] = nn.Linear(size, d_model)
        
        # Pre-allocate context fusion layer with reasonable max size
        max_fusion_dim = d_model * 6  # Reasonable upper bound for fusion
        self.context_fusion_layer = nn.Linear(max_fusion_dim, d_model)
    
    def project_context_summary(self, summary: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Project an arbitrary summary tensor to d_model dimensions."""
        summary_dim = summary.size(-1)
        
        # Find the closest pre-allocated projector
        projector_key = f'proj_{summary_dim}'
        if projector_key in self.context_projection_layers:
            projector = self.context_projection_layers[projector_key]
            return projector(summary)
        
        # Find the next larger projector and pad input
        available_sizes = [int(k.split('_')[1]) for k in self.context_projection_layers.keys() if k.startswith('proj_')]
        larger_sizes = [s for s in available_sizes if s >= summary_dim]
        
        if larger_sizes:
            target_size = min(larger_sizes)
            projector_key = f'proj_{target_size}'
            projector = self.context_projection_layers[projector_key]
            
            # Pad input to match projector size
            if summary_dim < target_size:
                batch_size = summary.size(0)
                padding = torch.zeros(batch_size, target_size - summary_dim, 
                                    device=device, dtype=summary.dtype)
                summary_padded = torch.cat([summary, padding], dim=-1)
                return projector(summary_padded)
        
        # Fallback: use largest available projector and truncate input
        max_size = max(available_sizes)
        projector_key = f'proj_{max_size}'
        projector = self.context_projection_layers[projector_key]
        
        if summary_dim > max_size:
            summary_truncated = summary[:, :max_size]
            return projector(summary_truncated)
        
        return projector(summary)
    
    def fuse_context(self, fusion_input: torch.Tensor) -> torch.Tensor:
        """Fuse context with padding/truncation handling"""
        fusion_dim = fusion_input.size(-1)
        max_fusion_dim = self.context_fusion_layer.in_features
        batch_size = fusion_input.size(0)
        
        if fusion_dim > max_fusion_dim:
            # Truncate if input is too large
            fusion_input = fusion_input[:, :max_fusion_dim]
        elif fusion_dim < max_fusion_dim:
            # Pad if input is too small
            padding = torch.zeros(batch_size, max_fusion_dim - fusion_dim, 
                                device=fusion_input.device, dtype=fusion_input.dtype)
            fusion_input = torch.cat([fusion_input, padding], dim=-1)

        return self.context_fusion_layer(fusion_input)


class PatchConfigGenerator:
    """Generates adaptive patch configurations"""
    
    @staticmethod
    def create_adaptive_patch_configs(seq_len: int) -> List[Dict[str, int]]:
        """Create patch configurations that are compatible with the sequence length."""
        configs = []
        
        # Ensure patch lengths don't exceed sequence length
        max_patch_len = seq_len // 2  # At most half the sequence length
        
        if seq_len >= 8:
            # Small patches (fine-grained)
            patch_len = min(4, max_patch_len)
            if patch_len >= 2:
                configs.append({'patch_len': patch_len, 'stride': max(1, patch_len // 2)})
        
        if seq_len >= 12:
            # Medium patches
            patch_len = min(8, max_patch_len)
            if patch_len >= 4:
                configs.append({'patch_len': patch_len, 'stride': max(2, patch_len // 2)})
        
        if seq_len >= 16:
            # Large patches (coarse-grained)
            patch_len = min(12, max_patch_len)
            if patch_len >= 6:
                configs.append({'patch_len': patch_len, 'stride': max(3, patch_len // 2)})
        
        # Fallback: if no configs generated, create a minimal one
        if not configs:
            patch_len = max(1, seq_len // 3)
            stride = max(1, patch_len // 2)
            configs.append({'patch_len': patch_len, 'stride': stride})
        
        return configs