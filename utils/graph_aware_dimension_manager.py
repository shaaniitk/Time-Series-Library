import dataclasses
from typing import List, Dict, Any, Optional, Tuple
import torch
from utils.dimension_manager import DimensionManager

@dataclasses.dataclass
class GraphAwareDimensionManager(DimensionManager):
    """
    Enhanced dimension manager that predicts and validates all graph-related dimensions
    upfront to prevent index out of bounds errors and dimension mismatches.
    
    This manager handles:
    - Node count calculations for different graph components
    - Edge index bounds validation
    - Spatial-temporal tensor shape compatibility
    - Batch dimension propagation through graph layers
    """
    
    # Core sequence dimensions
    seq_len: int = 96
    pred_len: int = 24
    d_model: int = 512
    
    # Graph-specific dimensions
    num_nodes: int = 8  # Default spatial nodes
    max_edge_index: int = None  # Will be calculated
    
    # Temporal segmentation
    wave_len: int = None  # Will be derived from seq_len
    target_len: int = None  # Will be derived from pred_len
    transition_len: int = None  # Overlap between wave and target
    
    def __post_init__(self):
        """Calculate all derived dimensions and validate compatibility."""
        super().__post_init__()
        
        # Calculate temporal segments
        self.wave_len = self.seq_len
        self.target_len = self.pred_len
        self.transition_len = min(self.wave_len, self.target_len)
        
        # Calculate graph dimensions
        self._calculate_graph_dimensions()
        
        # Validate all dimensions
        self._validate_dimensions()
    
    def _calculate_graph_dimensions(self):
        """Calculate all graph-related dimensions."""
        # Node counts for different graph components
        self.node_counts = {
            'wave': self.wave_len,
            'target': self.target_len,
            'transition': self.transition_len,
            'spatial': self.num_nodes
        }
        
        # Edge index bounds for different graph types
        self.edge_bounds = {
            ('wave', 'transition'): (self.wave_len, self.transition_len),
            ('transition', 'target'): (self.transition_len, self.target_len),
            ('wave', 'target'): (self.wave_len, self.target_len),
            'spatial': (self.num_nodes, self.num_nodes)
        }
        
        # Calculate maximum edge index for validation
        self.max_edge_index = max(
            max(src, tgt) for src, tgt in self.edge_bounds.values()
        )
        
        # Spatial-temporal tensor shapes
        self.tensor_shapes = {
            'embedded': (None, self.seq_len + self.pred_len, self.d_model),  # batch varies
            'wave_embedded': (None, self.wave_len, self.d_model),
            'target_embedded': (None, self.target_len, self.d_model),
            'spatiotemporal': (None, self.seq_len + self.pred_len, self.num_nodes, self.d_model),
            'node_features': {
                'wave': (self.wave_len, self.d_model),
                'target': (self.target_len, self.d_model),
                'transition': (self.transition_len, self.d_model)
            }
        }
    
    def _validate_dimensions(self):
        """Validate all dimension calculations for consistency."""
        issues = []
        
        # Validate temporal segments
        if self.wave_len <= 0:
            issues.append(f"Invalid wave_len: {self.wave_len}")
        if self.target_len <= 0:
            issues.append(f"Invalid target_len: {self.target_len}")
        if self.transition_len <= 0:
            issues.append(f"Invalid transition_len: {self.transition_len}")
        
        # Validate node counts
        for node_type, count in self.node_counts.items():
            if count <= 0:
                issues.append(f"Invalid node count for {node_type}: {count}")
        
        # Validate edge bounds
        for edge_type, (src, tgt) in self.edge_bounds.items():
            if src <= 0 or tgt <= 0:
                issues.append(f"Invalid edge bounds for {edge_type}: ({src}, {tgt})")
        
        if issues:
            raise ValueError(f"Dimension validation failed: {'; '.join(issues)}")
    
    def get_edge_index_shape(self, edge_type: str) -> Tuple[int, int]:
        """Get the expected shape for edge indices of a given type."""
        if edge_type not in self.edge_bounds:
            raise ValueError(f"Unknown edge type: {edge_type}")
        
        src_nodes, tgt_nodes = self.edge_bounds[edge_type]
        # For fully connected bipartite graph: [2, src_nodes * tgt_nodes]
        return (2, src_nodes * tgt_nodes)
    
    def validate_edge_index(self, edge_index: torch.Tensor, edge_type: str) -> bool:
        """Validate that edge indices are within bounds."""
        if edge_type not in self.edge_bounds:
            return False
        
        src_max, tgt_max = self.edge_bounds[edge_type]
        
        # Check shape
        expected_shape = self.get_edge_index_shape(edge_type)
        if edge_index.shape != expected_shape:
            return False
        
        # Check bounds
        src_indices = edge_index[0]
        tgt_indices = edge_index[1]
        
        if torch.any(src_indices >= src_max) or torch.any(src_indices < 0):
            return False
        if torch.any(tgt_indices >= tgt_max) or torch.any(tgt_indices < 0):
            return False
        
        return True
    
    def get_node_feature_shape(self, node_type: str, batch_size: int = None) -> Tuple[int, ...]:
        """Get expected shape for node features of a given type."""
        if node_type not in self.node_counts:
            raise ValueError(f"Unknown node type: {node_type}")
        
        node_count = self.node_counts[node_type]
        if batch_size is not None:
            return (batch_size, node_count, self.d_model)
        else:
            return (node_count, self.d_model)
    
    def validate_tensor_shape(self, tensor: torch.Tensor, expected_type: str, batch_size: int = None) -> bool:
        """Validate that a tensor has the expected shape."""
        if expected_type not in self.tensor_shapes:
            return False
        
        expected_shape = self.tensor_shapes[expected_type]
        
        # Handle batch dimension
        if batch_size is not None and expected_shape[0] is None:
            expected_shape = (batch_size,) + expected_shape[1:]
        
        return tensor.shape == expected_shape
    
    def create_safe_edge_index(self, src_nodes: int, tgt_nodes: int, device: torch.device) -> torch.Tensor:
        """Create edge indices with built-in bounds checking."""
        # Validate input bounds
        if src_nodes <= 0 or tgt_nodes <= 0:
            raise ValueError(f"Invalid node counts: src={src_nodes}, tgt={tgt_nodes}")
        
        if src_nodes > self.max_edge_index or tgt_nodes > self.max_edge_index:
            raise ValueError(f"Node counts exceed maximum: src={src_nodes}, tgt={tgt_nodes}, max={self.max_edge_index}")
        
        # Create fully connected bipartite graph
        src_indices = torch.arange(src_nodes, device=device).repeat(tgt_nodes)
        tgt_indices = torch.arange(tgt_nodes, device=device).repeat_interleave(src_nodes)
        edge_index = torch.stack([src_indices, tgt_indices], dim=0)
        
        return edge_index
    
    def get_graph_component_info(self) -> Dict[str, Any]:
        """Get comprehensive information about graph components."""
        return {
            'temporal_segments': {
                'wave_len': self.wave_len,
                'target_len': self.target_len,
                'transition_len': self.transition_len,
                'total_seq_len': self.seq_len + self.pred_len
            },
            'node_counts': self.node_counts.copy(),
            'edge_bounds': self.edge_bounds.copy(),
            'max_edge_index': self.max_edge_index,
            'tensor_shapes': self.tensor_shapes.copy()
        }
    
    def validate_graph_compatibility(self, components: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate that graph components are compatible with dimensions."""
        issues = []
        
        # Check if components require graph features
        graph_components = ['enhanced_pgat', 'dynamic_graph', 'graph_attention', 'spatial_encoder']
        
        for comp_name, comp_config in components.items():
            if comp_name in graph_components:
                # Validate graph-specific requirements
                if hasattr(comp_config, 'num_nodes'):
                    if comp_config.num_nodes != self.num_nodes:
                        issues.append(f"{comp_name} expects {comp_config.num_nodes} nodes, but manager has {self.num_nodes}")
                
                if hasattr(comp_config, 'd_model'):
                    if comp_config.d_model != self.d_model:
                        issues.append(f"{comp_name} expects d_model={comp_config.d_model}, but manager has {self.d_model}")
        
        return len(issues) == 0, issues
    
    def __repr__(self):
        return (
            f"GraphAwareDimensionManager(\n"
            f"  Temporal: wave={self.wave_len}, target={self.target_len}, transition={self.transition_len}\n"
            f"  Spatial: nodes={self.num_nodes}, max_edge_idx={self.max_edge_index}\n"
            f"  Model: d_model={self.d_model}, seq_len={self.seq_len}, pred_len={self.pred_len}\n"
            f"  Features: {self.num_targets} targets, {self.num_covariates} covariates\n"
            f")"
        )

def create_graph_aware_dimension_manager(config) -> GraphAwareDimensionManager:
    """Factory function to create GraphAwareDimensionManager from config."""
    return GraphAwareDimensionManager(
        mode=getattr(config, 'features', 'MS'),
        target_features=getattr(config, 'target_features', ['target']),
        all_features=getattr(config, 'all_features', ['target'] + [f'cov_{i}' for i in range(getattr(config, 'enc_in', 7) - 1)]),
        loss_function=getattr(config, 'loss_function_type', 'mse'),
        quantiles=getattr(config, 'quantile_levels', []),
        seq_len=getattr(config, 'seq_len', 96),
        pred_len=getattr(config, 'pred_len', 24),
        d_model=getattr(config, 'd_model', 512),
        num_nodes=getattr(config, 'num_nodes', 8)
    )

if __name__ == "__main__":
    # Test the enhanced dimension manager
    from argparse import Namespace
    
    test_config = Namespace(
        features='MS',
        enc_in=10,
        dec_in=10,
        c_out=7,
        seq_len=96,
        pred_len=24,
        d_model=512,
        num_nodes=8,
        loss_function_type='mse',
        quantile_levels=[]
    )
    
    dim_manager = create_graph_aware_dimension_manager(test_config)
    print("=== Graph-Aware Dimension Manager ===")
    print(dim_manager)
    
    print("\n=== Graph Component Info ===")
    info = dim_manager.get_graph_component_info()
    for category, details in info.items():
        print(f"{category}: {details}")
    
    print("\n=== Edge Index Validation Test ===")
    device = torch.device('cpu')
    
    # Test safe edge index creation
    try:
        edge_index = dim_manager.create_safe_edge_index(96, 24, device)
        print(f"Created edge index shape: {edge_index.shape}")
        
        # Validate the created edge index
        is_valid = dim_manager.validate_edge_index(edge_index, ('wave', 'target'))
        print(f"Edge index validation: {is_valid}")
        
    except Exception as e:
        print(f"Edge index creation failed: {e}")
    
    print("\n=== Tensor Shape Validation Test ===")
    batch_size = 32
    test_tensor = torch.randn(batch_size, 120, 512)  # seq_len + pred_len = 96 + 24 = 120
    is_valid = dim_manager.validate_tensor_shape(test_tensor, 'embedded', batch_size)
    print(f"Embedded tensor validation: {is_valid}")