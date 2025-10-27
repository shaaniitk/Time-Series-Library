"""
Graph utilities for Enhanced SOTA PGAT
Contains graph validation, edge processing, and proposal management
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union


class GraphValidator:
    """Validates graph component outputs and provides robust fallbacks"""
    
    @staticmethod
    def validate_graph_output(graph_result: Any, component_name: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Validate graph component output format and provide robust fallbacks."""
        if isinstance(graph_result, (tuple, list)):
            if len(graph_result) >= 2:
                return graph_result[0], graph_result[1]
            elif len(graph_result) == 1:
                print(f"Warning: {component_name} returned single element tuple, using None for weights")
                return graph_result[0], None
            else:
                raise ValueError(f"Invalid tuple length from {component_name}: {len(graph_result)}")
        elif isinstance(graph_result, torch.Tensor):
            return graph_result, None
        else:
            raise ValueError(f"Invalid output format from {component_name}: {type(graph_result)}")


class EdgeProcessor:
    """Processes edge indices and weights for graph attention"""
    
    @staticmethod
    def normalize_edge_batch_outputs(adjacency_result: Any, batch_size: int) -> Tuple[
        List[Dict[Tuple[str, str, str], torch.Tensor]], 
        List[Optional[Dict[Tuple[str, str, str], torch.Tensor]]]
    ]:
        """Coerce adjacency_to_edge_indices output into per-batch lists."""
        if isinstance(adjacency_result, tuple):
            edge_indices_raw, edge_weights_raw = adjacency_result
        else:
            edge_indices_raw = adjacency_result
            edge_weights_raw = None

        if isinstance(edge_indices_raw, dict):
            edge_indices = [edge_indices_raw]
        else:
            edge_indices = list(edge_indices_raw)

        if edge_weights_raw is None:
            edge_weights: List[Optional[Dict[Tuple[str, str, str], torch.Tensor]]] = [None] * len(edge_indices)
        elif isinstance(edge_weights_raw, dict):
            edge_weights = [edge_weights_raw]
        else:
            edge_weights = list(edge_weights_raw)

        if len(edge_indices) != batch_size:
            if len(edge_indices) == 1:
                edge_indices = edge_indices * batch_size
                edge_weights = edge_weights * batch_size
            else:
                raise ValueError(
                    f"Edge index batch size {len(edge_indices)} does not match input batch size {batch_size}."
                )

        return edge_indices, edge_weights


class GraphProposalManager:
    """Manages multiple graph proposals for gated combination"""
    
    def __init__(self):
        self.proposals = []
    
    def add_proposal(self, adjacency: torch.Tensor, weights: Optional[torch.Tensor], 
                    batch_size: int, total_nodes: int, name: str = ""):
        """Add a graph proposal to the collection"""
        # This would call prepare_graph_proposal utility function
        # For now, storing the raw components
        proposal = {
            'adjacency': adjacency,
            'weights': weights,
            'batch_size': batch_size,
            'total_nodes': total_nodes,
            'name': name
        }
        self.proposals.append(proposal)
    
    def get_proposals(self) -> List[Any]:
        """Get all proposals in the format expected by GatedGraphCombiner"""
        # This would return properly formatted proposals
        # Implementation depends on the prepare_graph_proposal function
        return self.proposals
    
    def clear(self):
        """Clear all proposals"""
        self.proposals.clear()


class GraphAttentionProcessor:
    """Processes graph attention for batched inputs"""
    
    @staticmethod
    def process_batch_attention(enhanced_x_dict: Dict[str, torch.Tensor], 
                              edge_index_batches: List[Any], 
                              edge_weight_batches: List[Any],
                              graph_attention_module: nn.Module,
                              batch_size: int) -> Dict[str, torch.Tensor]:
        """Process graph attention for each batch item"""
        batch_attended_features = {'wave': [], 'transition': [], 'target': []}

        for b in range(batch_size):
            single_batch_x_dict = {
                'wave': enhanced_x_dict['wave'][b],
                'transition': enhanced_x_dict['transition'][b],
                'target': enhanced_x_dict['target'][b],
            }
            edge_index_single = edge_index_batches[b]
            edge_weight_single = edge_weight_batches[b]

            attended_single = graph_attention_module(
                single_batch_x_dict,
                edge_index_single,
                edge_weight_single,
            )

            for key in batch_attended_features:
                batch_attended_features[key].append(attended_single[key])

        spatial_encoded = {
            'wave': torch.stack(batch_attended_features['wave'], dim=0),
            'transition': torch.stack(batch_attended_features['transition'], dim=0),
            'target': torch.stack(batch_attended_features['target'], dim=0),
        }
        
        return spatial_encoded


# Placeholder functions that would need to be imported from existing modules
def prepare_graph_proposal(adjacency, weights, batch_size, total_nodes, preserve_weights=True):
    """
    Placeholder for the prepare_graph_proposal function
    This should be imported from the appropriate graph utilities module
    """
    # This is a placeholder - the actual implementation should be imported
    return (adjacency, weights)


def validate_graph_proposals(graph_proposals, batch_size, total_nodes):
    """
    Placeholder for the validate_graph_proposals function
    This should be imported from the appropriate graph utilities module
    """
    # This is a placeholder - the actual implementation should be imported
    return len(graph_proposals) > 0


def adjacency_to_edge_indices(adjacency_matrix, wave_nodes, target_nodes, transition_nodes, edge_weights=None):
    """
    Placeholder for the adjacency_to_edge_indices function
    This should be imported from the appropriate graph utilities module
    """
    # This is a placeholder - the actual implementation should be imported
    return (adjacency_matrix, edge_weights)