"""
Context attention mechanisms for Enhanced SOTA PGAT
Contains rich context creation and attention-based feature aggregation
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from ..features.phase_features import PatchAggregator


class ContextAttentionProcessor(nn.Module):
    """Processes context attention for rich feature aggregation"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.context_attention = nn.Linear(d_model, 1)
    
    def create_rich_context(self, all_node_features: torch.Tensor, 
                           wave_patched_outputs: Optional[Dict[str, torch.Tensor]] = None,
                           target_patched_outputs: Optional[Dict[str, torch.Tensor]] = None,
                           projection_manager=None) -> torch.Tensor:
        """
        Create rich context tensor that preserves multi-scale and stochastic information
        instead of simple averaging that loses important features.
        
        Args:
            all_node_features: [batch_size, total_nodes, d_model]
            wave_patched_outputs: Dict of wave patch outputs (optional)
            target_patched_outputs: Dict of target patch outputs (optional)
            projection_manager: ProjectionManager instance for handling projections
            
        Returns:
            Rich context tensor [batch_size, d_model]
        """
        batch_size, total_nodes, d_model = all_node_features.shape

        # Attention-weighted pooling instead of simple mean
        attention_logits = self.context_attention(all_node_features)
        attention_weights = torch.softmax(attention_logits, dim=1)
        base_context = torch.sum(all_node_features * attention_weights, dim=1)

        # Statistical features
        node_mean = all_node_features.mean(dim=1)
        node_std = all_node_features.std(dim=1, unbiased=False)

        # Convert patch output dictionaries to lists of tensors
        wave_patch_list = list(wave_patched_outputs.values()) if wave_patched_outputs else []
        target_patch_list = list(target_patched_outputs.values()) if target_patched_outputs else []
        
        # Aggregate patch information
        wave_summary = PatchAggregator.aggregate_patch_collection(
            wave_patch_list,
            "wave_patch",
            batch_size,
            all_node_features.device,
            all_node_features.dtype,
            self.d_model,
            projection_manager
        )
        target_summary = PatchAggregator.aggregate_patch_collection(
            target_patch_list,
            "target_patch",
            batch_size,
            all_node_features.device,
            all_node_features.dtype,
            self.d_model,
            projection_manager
        )

        # Combine all context information
        fusion_input = torch.cat([base_context, node_mean, node_std, wave_summary, target_summary], dim=-1)

        # Use projection manager for safe fusion
        if projection_manager is not None:
            context_vector = projection_manager.fuse_context(fusion_input)
        else:
            # Fallback to simple linear projection
            context_vector = base_context  # Use base context as fallback
        
        return context_vector


class AttentionWeightExtractor:
    """Extracts and processes attention weights for interpretability"""
    
    @staticmethod
    def extract_relationship_matrix(attention_weights: torch.Tensor) -> torch.Tensor:
        """Extract interpretable covariate-target relationships"""
        if attention_weights is None:
            return None
        
        # Convert to [celestial_features x ohlc_targets] matrix
        relationship_matrix = attention_weights.mean(dim=0)  # Average over batch
        return relationship_matrix
    
    @staticmethod
    def get_top_relationships(relationship_matrix: torch.Tensor, k: int = 10) -> torch.Tensor:
        """Get top-k strongest relationships"""
        if relationship_matrix is None:
            return None
        
        top_indices = torch.topk(relationship_matrix.flatten(), k)
        return top_indices


class HierarchicalAttentionWrapper:
    """Wrapper for hierarchical temporal-spatial attention components"""
    
    def __init__(self, config, d_model: int):
        self.config = config
        self.d_model = d_model
        self.use_hierarchical_mapper = getattr(config, 'use_hierarchical_mapper', True)
        
        if self.use_hierarchical_mapper:
            # These would be imported from the appropriate modules
            # For now, we'll store the configuration
            self.wave_mapper_config = {
                'd_model': d_model,
                'num_nodes': getattr(config, 'num_wave_features', 4),
                'n_heads': getattr(config, 'n_heads', 8)
            }
            self.target_mapper_config = {
                'd_model': d_model,
                'num_nodes': getattr(config, 'c_out', 3),
                'n_heads': getattr(config, 'n_heads', 8)
            }
    
    def create_mappers(self):
        """Create hierarchical temporal-spatial mappers"""
        if not self.use_hierarchical_mapper:
            return None, None
        
        # This would import and create the actual HierarchicalTemporalSpatialMapper instances
        # For now, returning None as placeholder
        return None, None
    
    def process_temporal_to_spatial(self, wave_embedded: torch.Tensor, 
                                  target_embedded: torch.Tensor,
                                  wave_mapper=None, target_mapper=None) -> tuple:
        """Process temporal-to-spatial conversion"""
        if self.use_hierarchical_mapper and wave_mapper is not None and target_mapper is not None:
            wave_spatial = wave_mapper(wave_embedded)
            target_spatial = target_mapper(target_embedded)
        else:
            # Fallback to simple mean pooling
            wave_nodes = self.wave_mapper_config['num_nodes']
            target_nodes = self.target_mapper_config['num_nodes']
            wave_spatial = wave_embedded.mean(dim=1).unsqueeze(1).expand(-1, wave_nodes, -1)
            target_spatial = target_embedded.mean(dim=1).unsqueeze(1).expand(-1, target_nodes, -1)
        
        return wave_spatial, target_spatial