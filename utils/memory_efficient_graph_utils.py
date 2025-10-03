"""Memory-efficient graph utilities for PGAT models."""

import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional
import functools


@functools.lru_cache(maxsize=32)
def get_cached_topology_features(num_waves: int, num_targets: int) -> Dict[str, torch.Tensor]:
    """Cache topology features to avoid recomputation."""
    # Simple topology features without NetworkX overhead
    wave_topo_feats = []
    for i in range(num_waves):
        # Simplified degree and clustering (avoid NetworkX)
        degree_norm = min(1.0, (num_waves - 1) / max(1, num_waves))
        clustering = 1.0 if num_waves <= 2 else 0.5  # Simplified clustering
        wave_topo_feats.append([degree_norm, clustering])
    
    target_topo_feats = [[0.0, 0.0]] * num_targets
    
    return {
        'wave': torch.tensor(wave_topo_feats, dtype=torch.float32),
        'target': torch.tensor(target_topo_feats, dtype=torch.float32)
    }


def create_sparse_edge_index(src_nodes: int, tgt_nodes: int, 
                           connectivity_ratio: float = 0.3) -> torch.Tensor:
    """Create sparse edge connectivity instead of full connectivity."""
    max_edges = int(src_nodes * tgt_nodes * connectivity_ratio)
    
    # Random sparse connectivity
    src_indices = torch.randint(0, src_nodes, (max_edges,))
    tgt_indices = torch.randint(0, tgt_nodes, (max_edges,))
    
    # Remove duplicates
    edge_set = set(zip(src_indices.tolist(), tgt_indices.tolist()))
    unique_edges = list(edge_set)
    
    if len(unique_edges) == 0:
        # Fallback: at least one connection
        unique_edges = [(0, 0)]
    
    src_final = torch.tensor([e[0] for e in unique_edges], dtype=torch.long)
    tgt_final = torch.tensor([e[1] for e in unique_edges], dtype=torch.long)
    
    return torch.stack([src_final, tgt_final])


@functools.lru_cache(maxsize=16)
def get_memory_efficient_pyg_graph(num_waves: int, num_targets: int, 
                                 num_transitions: int, device: str) -> HeteroData:
    """Create memory-efficient PyG graph with caching."""
    data = HeteroData()
    
    # Define node counts
    data['wave'].num_nodes = num_waves
    data['target'].num_nodes = num_targets  
    data['transition'].num_nodes = num_transitions
    
    # Create sparse edge connectivity (much more memory efficient)
    wave_transition_edges = create_sparse_edge_index(num_waves, num_transitions, 0.5)
    transition_target_edges = create_sparse_edge_index(num_transitions, num_targets, 0.5)
    
    data['wave', 'interacts_with', 'transition'].edge_index = wave_transition_edges
    data['transition', 'influences', 'target'].edge_index = transition_target_edges
    
    # Get cached topology features
    topo_features = get_cached_topology_features(num_waves, num_targets)
    data['wave'].t = topo_features['wave']
    data['target'].t = topo_features['target']
    
    return data.to(device)


class MemoryEfficientGraphManager:
    """Manages graph structures with memory optimization."""
    
    def __init__(self, max_cache_size: int = 8):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_count = {}
    
    def get_graph(self, config, device: torch.device) -> HeteroData:
        """Get graph with LRU caching."""
        key = (config.num_waves, config.num_targets, config.num_transitions, str(device))
        
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        
        # Create new graph
        graph = get_memory_efficient_pyg_graph(
            config.num_waves, 
            config.num_targets,
            config.num_transitions,
            str(device)
        )
        
        # Cache management
        if len(self.cache) >= self.max_cache_size:
            # Remove least recently used
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = graph
        self.access_count[key] = 1
        
        return graph
    
    def clear_cache(self):
        """Clear all cached graphs."""
        self.cache.clear()
        self.access_count.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cached_graphs': len(self.cache),
            'total_accesses': sum(self.access_count.values()),
            'cache_hits': sum(1 for count in self.access_count.values() if count > 1)
        }


# Global graph manager instance
_graph_manager = MemoryEfficientGraphManager()


def get_pyg_graph(config, device: torch.device) -> HeteroData:
    """Memory-efficient replacement for original get_pyg_graph function."""
    return _graph_manager.get_graph(config, device)


def clear_graph_cache():
    """Clear global graph cache."""
    _graph_manager.clear_cache()


def get_graph_cache_stats() -> Dict[str, int]:
    """Get global graph cache statistics."""
    return _graph_manager.get_cache_stats()