"""
Graph utility functions for converting between different graph representations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple, Union, Optional, List, cast
from torch_geometric.data import HeteroData
import networkx as nx

# Export list for proper module discovery
__all__ = [
    'convert_hetero_to_dense_adj',
    'ensure_tensor_graph_format',
    'ensure_tensor_graph_format_with_weights',
    'adjacency_to_edge_indices',
    'create_petri_net_structure',
    'prepare_graph_proposal',
    'validate_learned_adjacency_structure',
    'validate_graph_proposals',
    'calculate_topology_features',
    'get_pyg_graph'
]

def convert_hetero_to_dense_adj(hetero_data: Union[HeteroData, torch.Tensor], 
                               total_nodes: Optional[int] = None,
                               preserve_weights: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert heterogeneous graph data to dense adjacency matrix.
    
    Args:
        hetero_data: HeteroData object or tensor
        total_nodes: Total number of nodes (inferred if None)
        
    Returns:
        Dense adjacency matrix [total_nodes, total_nodes]
    """
    if isinstance(hetero_data, torch.Tensor):
        # Already a tensor, return as-is
        return hetero_data
    
    if not isinstance(hetero_data, HeteroData):
        # Handle other types by returning identity matrix
        if total_nodes is None:
            total_nodes = 10  # Default fallback
        device = torch.device('cpu')
        return torch.eye(total_nodes, device=device)
    
    # Extract node counts from HeteroData
    node_counts = {}
    total_inferred = 0
    
    for node_type in hetero_data.node_types:
        if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
            count = hetero_data[node_type].x.size(0)
        elif hasattr(hetero_data[node_type], 'num_nodes'):
            count = hetero_data[node_type].num_nodes
        else:
            count = 1  # Fallback
        node_counts[node_type] = count
        total_inferred += count
    
    # Ensure the adjacency matrix can accommodate all inferred nodes.
    # If a smaller "total_nodes" was provided, use the maximum to avoid
    # out-of-bounds indexing when mapping hetero edges into the dense tensor.
    if total_nodes is None:
        total_nodes = total_inferred
    else:
        try:
            total_nodes = int(total_nodes)
        except Exception:
            # Fallback to inferred if casting fails
            total_nodes = total_inferred
        # Use the larger value to safely fit all indices
        total_nodes = max(total_nodes, int(total_inferred))
    
    assert isinstance(total_nodes, int), "total_nodes must be an integer at this point"

    # Create dense adjacency matrix
    device = next(iter(hetero_data.edge_stores)).edge_index.device if hetero_data.edge_types else torch.device('cpu')
    adj_matrix = torch.zeros(total_nodes, total_nodes, device=device)
    weight_matrix = torch.zeros(total_nodes, total_nodes, device=device) if preserve_weights else None
    
    # Map node types to index ranges
    node_type_ranges = {}
    start_idx = 0
    for node_type in ['wave', 'transition', 'target']:  # Fixed order
        if node_type in node_counts:
            end_idx = start_idx + node_counts[node_type]
            node_type_ranges[node_type] = (start_idx, end_idx)
            start_idx = end_idx
    
    # Fill adjacency matrix from edge indices
    for edge_type in hetero_data.edge_types:
        src_type, relation, dst_type = edge_type
        if src_type in node_type_ranges and dst_type in node_type_ranges:
            edge_index = hetero_data[edge_type].edge_index
            
            src_start, _ = node_type_ranges[src_type]
            dst_start, _ = node_type_ranges[dst_type]
            
            # Adjust indices to global range
            src_indices = edge_index[0] + src_start
            dst_indices = edge_index[1] + dst_start
            
            # Set edges in adjacency matrix
            adj_matrix[src_indices, dst_indices] = 1.0
            
            # Preserve edge weights if available
            if preserve_weights and weight_matrix is not None:
                edge_attr = getattr(hetero_data[edge_type], 'edge_attr', None)
                if edge_attr is not None:
                    if edge_attr.dim() == 1:
                        weight_matrix[src_indices, dst_indices] = edge_attr
                    else:
                        # Use first feature if multi-dimensional
                        weight_matrix[src_indices, dst_indices] = edge_attr[:, 0]
                else:
                    # Default weight of 1.0 for edges without explicit weights
                    weight_matrix[src_indices, dst_indices] = 1.0
    
    if preserve_weights and weight_matrix is not None:
        return adj_matrix, weight_matrix
    else:
        return adj_matrix


def ensure_tensor_graph_format(graph_data: Union[HeteroData, torch.Tensor, Dict], 
                              expected_nodes: int) -> torch.Tensor:
    """
    Ensure graph data is in tensor format for GatedGraphCombiner.
    
    Args:
        graph_data: Graph in various formats
        expected_nodes: Expected number of nodes
        
    Returns:
        Dense adjacency tensor [expected_nodes, expected_nodes]
    """
    result = ensure_tensor_graph_format_with_weights(graph_data, expected_nodes, preserve_weights=False)
    if isinstance(result, tuple):
        # Defensive guard – helper should not return tuple when preserve_weights=False
        adjacency_tensor, _ = result
        return adjacency_tensor
    return result


def ensure_tensor_graph_format_with_weights(graph_data: Union[HeteroData, torch.Tensor, Dict], 
                                          expected_nodes: int,
                                          preserve_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    Ensure graph data is in tensor format, optionally preserving weights.
    
    Args:
        graph_data: Graph in various formats
        expected_nodes: Expected number of nodes
        preserve_weights: Whether to return weights as well
        
    Returns:
        Dense adjacency tensor [expected_nodes, expected_nodes], or tuple (adjacency, weights) if preserve_weights=True
    """
    if isinstance(graph_data, torch.Tensor):
        # Handle different tensor dimensions
        if graph_data.dim() == 2:
            if graph_data.size(0) == expected_nodes and graph_data.size(1) == expected_nodes:
                result = graph_data
            else:
                # Resize 2D tensor
                device = graph_data.device
                new_adj = torch.zeros(expected_nodes, expected_nodes, device=device)
                min_size = min(graph_data.size(0), expected_nodes)
                new_adj[:min_size, :min_size] = graph_data[:min_size, :min_size]
                result = new_adj
        elif graph_data.dim() == 3:
            # Handle batched tensor - PRESERVE batch dimension instead of collapsing
            batch_size, nodes1, nodes2 = graph_data.shape
            if nodes1 == expected_nodes and nodes2 == expected_nodes:
                result = graph_data  # Return full batch
            else:
                # Resize all batches
                device = graph_data.device
                new_adj = torch.zeros(batch_size, expected_nodes, expected_nodes, device=device)
                min_size = min(nodes1, expected_nodes)
                new_adj[:, :min_size, :min_size] = graph_data[:, :min_size, :min_size]
                result = new_adj
        else:
            # Unknown tensor format, create identity
            result = torch.eye(expected_nodes, device=graph_data.device)
        
        # For tensor input, no separate weights available
        return (result, None) if preserve_weights else result
    
    elif isinstance(graph_data, HeteroData):
        # Handle HeteroData with proper weight preservation
        conversion_result = convert_hetero_to_dense_adj(
            graph_data,
            expected_nodes,
            preserve_weights=preserve_weights,
        )
        if preserve_weights:
            if isinstance(conversion_result, tuple):
                adjacency_tensor, weight_tensor = conversion_result
                return adjacency_tensor, weight_tensor
            return conversion_result, None
        if isinstance(conversion_result, tuple):
            adjacency_tensor, _ = conversion_result
            return adjacency_tensor
        return conversion_result
    
    elif isinstance(graph_data, dict):
        # Handle dictionary format (edge weights, etc.)
        device = torch.device('cpu')
        if 'adjacency' in graph_data:
            result = ensure_tensor_graph_format_with_weights(
                graph_data['adjacency'], expected_nodes, preserve_weights
            )
            if preserve_weights:
                return result
            if isinstance(result, tuple):
                adjacency_tensor, _ = result
                return adjacency_tensor
            return result
        else:
            # Create identity matrix as fallback
            identity = torch.eye(expected_nodes, device=device)
            return (identity, None) if preserve_weights else identity
    
    else:
        # Unknown format, return identity
        identity = torch.eye(expected_nodes)
        return (identity, None) if preserve_weights else identity


def adjacency_to_edge_indices(
    adjacency_matrix: torch.Tensor,
    wave_nodes: int,
    target_nodes: int,
    transition_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
    threshold: float = 0.1,
    batch_idx: Optional[int] = None
) -> Union[
    Dict[Tuple[str, str, str], torch.Tensor],
    List[Dict[Tuple[str, str, str], torch.Tensor]],
    Tuple[List[Dict[Tuple[str, str, str], torch.Tensor]], List[Dict[Tuple[str, str, str], torch.Tensor]]],
    Tuple[Dict[Tuple[str, str, str], torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]
]:
    """Convert a learned adjacency tensor into per-sample heterogeneous edge indices.

    The returned edge dictionaries follow the PyG hetero convention and preserve
    batch-specific structure instead of collapsing to the first sample.

    Args:
        adjacency_matrix: Learned adjacency of shape ``[B, N, N]`` or ``[N, N]``.
        wave_nodes: Number of wave nodes present in the graph.
        target_nodes: Number of target nodes present in the graph.
        transition_nodes: Number of transition nodes present in the graph.
        edge_weights: Optional tensor of the same shape as ``adjacency_matrix``
            describing learned edge weights.
        threshold: Minimum edge strength to retain when building edge indices.
        batch_idx: Optional batch index to extract a single sample. When ``None``
            (default) all samples are converted individually.

    Returns:
        Either a list of edge-index dictionaries (one per batch item) or a tuple
        of lists ``(edge_indices, edge_weights)`` when ``edge_weights`` are
        supplied.
    """

    total_nodes = wave_nodes + transition_nodes + target_nodes

    if adjacency_matrix.dim() == 2:
        adjacency_matrix = adjacency_matrix.unsqueeze(0)
        if edge_weights is not None:
            edge_weights = edge_weights.unsqueeze(0)

    if adjacency_matrix.dim() != 3:
        raise ValueError(
            f"adjacency_matrix must have shape [batch, N, N] or [N, N]; got {adjacency_matrix.shape}"
        )

    batch_indices: List[int]
    batch_size = adjacency_matrix.size(0)
    if batch_idx is not None:
        if batch_idx < 0 or batch_idx >= batch_size:
            raise IndexError(f"batch_idx {batch_idx} is out of range for batch size {batch_size}")
        batch_indices = [batch_idx]
    else:
        batch_indices = list(range(batch_size))

    wave_start, wave_end = 0, wave_nodes
    transition_start, transition_end = wave_nodes, wave_nodes + transition_nodes
    target_start, target_end = wave_nodes + transition_nodes, total_nodes

    edge_index_batch: List[Dict[Tuple[str, str, str], torch.Tensor]] = []
    edge_weight_batch: List[Dict[Tuple[str, str, str], torch.Tensor]] = [] if edge_weights is not None else []

    for b_idx in batch_indices:
        adj_sample = adjacency_matrix[b_idx]
        if adj_sample.shape != (total_nodes, total_nodes):
            raise ValueError(
                f"Adjacency matrix shape {adj_sample.shape} does not match expected {(total_nodes, total_nodes)}"
            )

        weight_sample = edge_weights[b_idx] if edge_weights is not None else None

        wave_to_transition = adj_sample[wave_start:wave_end, transition_start:transition_end]
        wave_mask = wave_to_transition > threshold
        wave_trans_edges = wave_mask.nonzero(as_tuple=False).t()

        if wave_trans_edges.numel() == 0:
            wave_trans_edges = torch.tensor([[0], [0]], device=adj_sample.device)

        trans_to_target = adj_sample[transition_start:transition_end, target_start:target_end]
        trans_mask = trans_to_target > threshold
        trans_target_edges = trans_mask.nonzero(as_tuple=False).t()

        if trans_target_edges.numel() == 0:
            trans_target_edges = torch.tensor([[0], [0]], device=adj_sample.device)

        edge_index_dict = {
            ('wave', 'interacts_with', 'transition'): wave_trans_edges.flip(0),
            ('transition', 'influences', 'target'): trans_target_edges.flip(0)
        }
        edge_index_batch.append(edge_index_dict)

        if weight_sample is not None:
            wave_weight_section = weight_sample[wave_start:wave_end, transition_start:transition_end]
            if wave_mask.any():
                wave_weights = wave_weight_section[wave_trans_edges[0], wave_trans_edges[1]]
            else:
                wave_weights = torch.ones(1, device=adj_sample.device)

            trans_weight_section = weight_sample[transition_start:transition_end, target_start:target_end]
            if trans_mask.any():
                trans_weights = trans_weight_section[trans_target_edges[0], trans_target_edges[1]]
            else:
                trans_weights = torch.ones(1, device=adj_sample.device)

            edge_weight_batch.append({
                ('wave', 'interacts_with', 'transition'): wave_weights,
                ('transition', 'influences', 'target'): trans_weights
            })

    if edge_weights is not None:
        if batch_idx is not None:
            return edge_index_batch[0], edge_weight_batch[0]
        return edge_index_batch, edge_weight_batch
    if batch_idx is not None:
        return edge_index_batch[0]
    return edge_index_batch


def create_petri_net_structure(wave_nodes: int, target_nodes: int, transition_nodes: int) -> torch.Tensor:
    """
    Create a Petri net-style adjacency matrix structure.
    
    Args:
        wave_nodes: Number of wave (input) nodes
        target_nodes: Number of target (output) nodes  
        transition_nodes: Number of transition (processing) nodes
        
    Returns:
        Adjacency matrix with Petri net structure
    """
    total_nodes = wave_nodes + transition_nodes + target_nodes
    adj_matrix = torch.zeros(total_nodes, total_nodes)
    
    # Wave nodes connect to transition nodes
    for i in range(wave_nodes):
        for j in range(transition_nodes):
            adj_matrix[i, wave_nodes + j] = 1.0
    
    # Transition nodes connect to target nodes  
    for i in range(transition_nodes):
        for j in range(target_nodes):
            adj_matrix[wave_nodes + i, wave_nodes + transition_nodes + j] = 1.0
    
    return adj_matrix


def prepare_graph_proposal(adjacency: Union[torch.Tensor, HeteroData, Dict], 
                          weights: Optional[Union[torch.Tensor, Dict]], 
                          batch_size: int, 
                          total_nodes: int,
                          preserve_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Prepare a graph proposal for the gated combiner by ensuring proper format and batch dimensions.
    
    Args:
        adjacency: Graph adjacency in various formats
        weights: Optional edge weights
        batch_size: Target batch size
        total_nodes: Expected number of nodes
        
    Returns:
        Tuple of (batched_adjacency, batched_weights) ready for gated combiner
    """
    # Convert to tensor format with weight preservation
    if isinstance(adjacency, HeteroData) and preserve_weights:
        conversion_result = convert_hetero_to_dense_adj(adjacency, total_nodes, preserve_weights=True)
        if isinstance(conversion_result, tuple):
            adj_tensor, extracted_weights = conversion_result
            # Normalize adjacency and weights to expected size
            normalized_adj, normalized_w = ensure_tensor_graph_format_with_weights(adj_tensor, total_nodes, preserve_weights=True)
            adj_tensor = normalized_adj
            if weights is None:
                weights = normalized_w if normalized_w is not None else extracted_weights
            else:
                # If explicit weights provided, also normalize them to expected size
                if isinstance(weights, torch.Tensor):
                    _, weights = ensure_tensor_graph_format_with_weights(weights, total_nodes, preserve_weights=True)
        else:
            # Normalize adjacency to expected size
            adj_tensor = ensure_tensor_graph_format(adjacency=conversion_result, expected_nodes=total_nodes)
    elif isinstance(adjacency, torch.Tensor) and adjacency.dim() >= 2:
        # Handle tensor adjacency - preserve as-is
        adj_tensor = ensure_tensor_graph_format(adjacency, total_nodes)
        # If weights is a tensor, preserve it
        if isinstance(weights, torch.Tensor):
            weights = ensure_tensor_graph_format(weights, total_nodes)
    else:
        adj_tensor = ensure_tensor_graph_format(adjacency, total_nodes)
    
    # Ensure batch dimension for adjacency
    if adj_tensor.dim() == 2:
        adj_tensor = adj_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    elif adj_tensor.dim() == 3 and adj_tensor.size(0) != batch_size:
        # Expand or contract to match batch size (preserve per-sample variation when possible)
        if adj_tensor.size(0) == 1:
            adj_tensor = adj_tensor.expand(batch_size, -1, -1)
        else:
            # Take first batch_size samples
            adj_tensor = adj_tensor[:batch_size]
    
    # Handle weights with proper batching
    weights_tensor = None
    if weights is not None:
        if isinstance(weights, torch.Tensor):
            weights_tensor = weights
            # Ensure batch dimension for weights
            if weights_tensor.dim() == 2:
                weights_tensor = weights_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            elif weights_tensor.dim() == 3 and weights_tensor.size(0) != batch_size:
                if weights_tensor.size(0) == 1:
                    weights_tensor = weights_tensor.expand(batch_size, -1, -1)
                else:
                    weights_tensor = weights_tensor[:batch_size]
        elif isinstance(weights, dict):
            # Convert dict weights to dense tensor (simplified approach)
            weights_tensor = torch.ones_like(adj_tensor)  # Default to uniform weights
    
    return adj_tensor, weights_tensor


def validate_learned_adjacency_structure(adjacency_matrix: torch.Tensor,
                                       wave_nodes: int,
                                       transition_nodes: int, 
                                       target_nodes: int) -> Dict[str, Any]:
    """
    Validate that learned adjacency matrix has reasonable structure for PGAT.
    
    Args:
        adjacency_matrix: Learned adjacency matrix
        wave_nodes: Number of wave nodes
        transition_nodes: Number of transition nodes
        target_nodes: Number of target nodes
        
    Returns:
        Dictionary with validation results and statistics
    """
    if adjacency_matrix.dim() == 3:
        adj = adjacency_matrix[0]  # Use first batch
    else:
        adj = adjacency_matrix
        
    total_nodes = wave_nodes + transition_nodes + target_nodes
    
    # Check basic shape
    shape_valid = adj.shape == (total_nodes, total_nodes)
    
    # Define ranges
    wave_range = slice(0, wave_nodes)
    trans_range = slice(wave_nodes, wave_nodes + transition_nodes)
    target_range = slice(wave_nodes + transition_nodes, total_nodes)
    
    # Extract submatrices
    wave_to_trans = adj[wave_range, trans_range]
    trans_to_target = adj[trans_range, target_range]
    
    # Count edges above threshold
    threshold = 0.1
    wave_trans_edges = (wave_to_trans > threshold).sum().item()
    trans_target_edges = (trans_to_target > threshold).sum().item()
    
    # Check for reasonable connectivity
    min_expected_edges = min(wave_nodes, transition_nodes)  # At least some connectivity
    
    validation_results = {
        'shape_valid': shape_valid,
        'expected_shape': (total_nodes, total_nodes),
        'actual_shape': adj.shape,
        'wave_to_transition_edges': wave_trans_edges,
        'transition_to_target_edges': trans_target_edges,
        'min_expected_edges': min_expected_edges,
        'has_sufficient_connectivity': wave_trans_edges >= 1 and trans_target_edges >= 1,
        'adjacency_stats': {
            'min': adj.min().item(),
            'max': adj.max().item(),
            'mean': adj.mean().item(),
            'std': adj.std().item()
        }
    }
    
    return validation_results


def validate_graph_proposals(proposals: List[Tuple[torch.Tensor, Optional[torch.Tensor]]], 
                           batch_size: int, 
                           total_nodes: int) -> bool:
    """
    Validate that all graph proposals have consistent shapes for gated combination.
    
    Args:
        proposals: List of (adjacency, weights) tuples
        batch_size: Expected batch size
        total_nodes: Expected number of nodes
        
    Returns:
        True if all proposals are valid, False otherwise
    """
    if not proposals:
        return False
    
    for i, (adj, weights) in enumerate(proposals):
        # Check adjacency matrix shape
        if adj.shape != (batch_size, total_nodes, total_nodes):
            print(f"Warning: Proposal {i} adjacency shape {adj.shape} != expected {(batch_size, total_nodes, total_nodes)}")
            return False
        
        # Check weights shape if present
        if weights is not None:
            if weights.dim() == 3 and weights.shape[0] != batch_size:
                print(f"Warning: Proposal {i} weights batch dimension {weights.shape[0]} != expected {batch_size}")
                return False
    
    return True

def calculate_topology_features(data: HeteroData) -> Dict[str, torch.Tensor]:
    """
    Calculates topology features (degree centrality, clustering coefficient) for wave nodes.
    Targets are treated as sinks with default zero-valued topology features.
    """
    num_waves = data['wave'].num_nodes
    
    # Create a graph for the 'wave' nodes
    # Assuming wave nodes are fully connected for topological feature calculation
    G = nx.complete_graph(num_waves)
    
    degrees: Dict[int, int] = dict(G.degree())
    clustering_coeffs: Dict[int, float] = cast(Dict[int, float], nx.clustering(G))
    
    max_degree = float(max(degrees.values())) if degrees else 1.0
    
    wave_topo_feats = []
    for i in range(num_waves):
        deg = degrees.get(i, 0) / max_degree
        cc = clustering_coeffs.get(i, 0.0)
        wave_topo_feats.append([deg, cc])
        
    # Targets have default zero features
    num_targets = data['target'].num_nodes if 'target' in data.node_types else 0
    target_topo_feats = [[0.0, 0.0]] * num_targets
            
    return {
        'wave': torch.tensor(wave_topo_feats, dtype=torch.float),
        'target': torch.tensor(target_topo_feats, dtype=torch.float)
    }

def get_pyg_graph(config, device: str) -> HeteroData:
    """Creates the complete PyG HeteroData object, including topology features."""
    data = HeteroData()

    # Using config for num_nodes
    num_waves = getattr(config, 'num_waves', 7)
    num_targets = getattr(config, 'num_targets', 3)
    num_transitions = getattr(config, 'num_transitions', min(num_waves, num_targets))

    data['wave'].num_nodes = num_waves
    data['target'].num_nodes = num_targets
    data['transition'].num_nodes = num_transitions

    # Edges from wave to transition
    wave_idx = torch.arange(num_waves).repeat_interleave(num_transitions)
    trans_idx_in = torch.arange(num_transitions).repeat(num_waves)
    data['wave', 'interacts_with', 'transition'].edge_index = torch.stack([wave_idx, trans_idx_in])

    # Edges from transition to target
    trans_idx_out = torch.arange(num_transitions).repeat_interleave(num_targets)
    target_idx = torch.arange(num_targets).repeat(num_transitions)
    data['transition', 'influences', 'target'].edge_index = torch.stack([trans_idx_out, target_idx])
    
    # Calculate and add topology features
    topo_features = calculate_topology_features(data)
    data['wave'].t = topo_features['wave']
    data['target'].t = topo_features['target']
    
    return data.to(device)
