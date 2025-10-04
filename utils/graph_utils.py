# File: 0_graph_utils.py

import torch
import networkx as nx
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx, from_networkx

def calculate_topology_features(data: HeteroData) -> dict:
    """
    Calculates topology features (degree centrality, clustering coefficient) for wave nodes.
    Targets are treated as sinks with default zero-valued topology features.
    """
    num_waves = data['wave'].num_nodes
    
    # We assume a fully-connected graph for waves to model all-to-all potential interactions.
    # This is a reasonable prior when the underlying physical connectivity is unknown or dense.
    G = nx.complete_graph(num_waves)
    
    # Calculate features using NetworkX
    degrees = dict(G.degree())
    clustering_coeffs = nx.clustering(G)
    
    # Normalize and stack features
    max_degree = float(max(degrees.values()) if degrees else 1.0)
    
    wave_topo_feats = []
    for i in range(num_waves):
        deg = degrees.get(i, 0) / max_degree
        cc = clustering_coeffs.get(i, 0.0)
        wave_topo_feats.append([deg, cc])
        
    target_topo_feats = [[0.0, 0.0]] * data['target'].num_nodes
            
    return {
        'wave': torch.tensor(wave_topo_feats, dtype=torch.float),
        'target': torch.tensor(target_topo_feats, dtype=torch.float)
    }

def get_pyg_graph(config, device: str) -> HeteroData:
    """Creates the complete PyG HeteroData object, including topology features."""
    data = HeteroData()

    # Define node types and their counts
    data['wave'].num_nodes = config.num_waves
    data['target'].num_nodes = config.num_targets
    data['transition'].num_nodes = config.num_transitions

    # Define edge types and their connectivity (the Petri Net structure)
    wave_idx = torch.arange(config.num_waves).repeat_interleave(config.num_transitions)
    trans_idx_in = torch.arange(config.num_transitions).repeat(config.num_waves)
    data['wave', 'interacts_with', 'transition'].edge_index = torch.stack([wave_idx, trans_idx_in])

    trans_idx_out = torch.arange(config.num_transitions).repeat_interleave(config.num_targets)
    target_idx = torch.arange(config.num_targets).repeat(config.num_transitions)
    data['transition', 'influences', 'target'].edge_index = torch.stack([trans_idx_out, target_idx])
    
    # Calculate and attach topology features as the '.t' attribute
    topo_features = calculate_topology_features(data)
    data['wave'].t = topo_features['wave']
    data['target'].t = topo_features['target']
    
    # Graph created successfully with topology features
    return data.to(device)