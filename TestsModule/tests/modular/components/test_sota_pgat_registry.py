"""Registry tests for SOTA_Temporal_PGAT components.

This module tests that all SOTA_Temporal_PGAT components are properly
registered in the modular framework and can be instantiated correctly.
"""

# Suppress optional dependency warnings
from TestsModule.suppress_warnings import suppress_optional_dependency_warnings
suppress_optional_dependency_warnings()

import pytest
import torch
from layers.modular.core.registry import ComponentRegistry, component_registry
# Import register_components to trigger component registration
import layers.modular.core.register_components


class TestSOTAPGATRegistry:
    """Test suite for SOTA_Temporal_PGAT component registration."""
    
    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Setup component registry for tests."""
        # Components are registered when register_components is imported
        self.registry = component_registry
    
    def test_multihead_graph_attention_registered(self):
        """Test that MultiHeadGraphAttention is properly registered."""
        component_name = "multihead_graph_attention"
        
        # Check if component is registered
        assert self.registry.is_registered(component_name), \
            f"Component {component_name} is not registered"
        
        # Test component instantiation
        config = {
            "d_model": 64,
            "n_heads": 8,
            "dropout": 0.1,
            "graph_dim": 32
        }
        
        component = self.registry.create_component(component_name, config)
        assert component is not None, "Failed to create MultiHeadGraphAttention component"
        
        # Test component has required attributes
        assert hasattr(component, 'forward'), "Component missing forward method"
        assert hasattr(component, 'd_model'), "Component missing d_model attribute"
        assert hasattr(component, 'n_heads'), "Component missing n_heads attribute"
    
    def test_graph_transformer_layer_registered(self):
        """Test that GraphTransformerLayer is properly registered."""
        component_name = "graph_transformer_layer"
        
        # Check if component is registered
        assert self.registry.is_registered(component_name), \
            f"Component {component_name} is not registered"
        
        # Test component instantiation
        config = {
            "d_model": 64,
            "n_heads": 8,
            "d_ff": 256,
            "dropout": 0.1,
            "graph_dim": 32
        }
        
        component = self.registry.create_component(component_name, config)
        assert component is not None, "Failed to create GraphTransformerLayer component"
    
    def test_graph_aware_positional_registered(self):
        """Test that GraphAwarePositionalEncoding is properly registered."""
        component_name = "graph_aware_positional"
        
        # Check if component is registered
        assert self.registry.is_registered(component_name), \
            f"Component {component_name} is not registered"
        
        # Test component instantiation
        config = {
            "d_model": 64,
            "max_len": 512,
            "graph_dim": 32,
            "dropout": 0.1
        }
        
        component = self.registry.create_component(component_name, config)
        assert component is not None, "Failed to create GraphAwarePositionalEncoding component"
    
    def test_hierarchical_graph_positional_registered(self):
        """Test that HierarchicalGraphPositionalEncoding is properly registered."""
        component_name = "hierarchical_graph_positional"
        
        # Check if component is registered
        assert self.registry.is_registered(component_name), \
            f"Component {component_name} is not registered"
        
        # Test component instantiation
        config = {
            "d_model": 64,
            "max_len": 512,
            "num_levels": 3,
            "graph_dim": 32,
            "dropout": 0.1
        }
        
        component = self.registry.create_component(component_name, config)
        assert component is not None, "Failed to create HierarchicalGraphPositionalEncoding component"
    
    def test_joint_spatiotemporal_registered(self):
        """Test that JointSpatioTemporalEncoding is properly registered."""
        component_name = "joint_spatiotemporal"
        
        # Check if component is registered
        assert self.registry.is_registered(component_name), \
            f"Component {component_name} is not registered"
        
        # Test component instantiation
        config = {
            "d_model": 64,
            "spatial_dim": 32,
            "temporal_dim": 32,
            "dropout": 0.1
        }
        
        component = self.registry.create_component(component_name, config)
        assert component is not None, "Failed to create JointSpatioTemporalEncoding component"
    
    def test_adaptive_spatiotemporal_registered(self):
        """Test that AdaptiveSpatioTemporalEncoder is properly registered."""
        component_name = "adaptive_spatiotemporal"
        
        # Check if component is registered
        assert self.registry.is_registered(component_name), \
            f"Component {component_name} is not registered"
        
        # Test component instantiation
        config = {
            "d_model": 64,
            "spatial_dim": 32,
            "temporal_dim": 32,
            "n_heads": 8,
            "dropout": 0.1
        }
        
        component = self.registry.create_component(component_name, config)
        assert component is not None, "Failed to create AdaptiveSpatioTemporalEncoder component"
    
    def test_dynamic_graph_constructor_registered(self):
        """Test that DynamicGraphConstructor is properly registered."""
        component_name = "dynamic_graph_constructor"
        
        # Check if component is registered
        assert self.registry.is_registered(component_name), \
            f"Component {component_name} is not registered"
        
        # Test component instantiation
        config = {
            "input_dim": 64,
            "hidden_dim": 32,
            "num_nodes": 10,
            "threshold": 0.5
        }
        
        component = self.registry.create_component(component_name, config)
        assert component is not None, "Failed to create DynamicGraphConstructor component"
    
    def test_adaptive_graph_structure_registered(self):
        """Test that AdaptiveGraphStructure is properly registered."""
        component_name = "adaptive_graph_structure"
        
        # Check if component is registered
        assert self.registry.is_registered(component_name), \
            f"Component {component_name} is not registered"
        
        # Test component instantiation
        config = {
            "input_dim": 64,
            "hidden_dim": 32,
            "num_nodes": 10,
            "adaptation_rate": 0.01
        }
        
        component = self.registry.create_component(component_name, config)
        assert component is not None, "Failed to create AdaptiveGraphStructure component"
    
    def test_all_sota_pgat_components_registered(self):
        """Test that all SOTA_Temporal_PGAT components are registered."""
        expected_components = [
            "multihead_graph_attention",
            "graph_transformer_layer",
            "graph_aware_positional",
            "hierarchical_graph_positional",
            "joint_spatiotemporal",
            "adaptive_spatiotemporal",
            "dynamic_graph_constructor",
            "adaptive_graph_structure"
        ]
        
        for component_name in expected_components:
            assert self.registry.is_registered(component_name), \
                f"SOTA_Temporal_PGAT component {component_name} is not registered"
    
    def test_component_families_correct(self):
        """Test that components are registered with correct families."""
        component_families = {
            "multihead_graph_attention": "attention",
            "graph_transformer_layer": "attention", 
            "graph_aware_positional": "embedding",
            "hierarchical_graph_positional": "embedding",
            "joint_spatiotemporal": "encoder",
            "adaptive_spatiotemporal": "encoder",
            "dynamic_graph_constructor": "graph",
            "adaptive_graph_structure": "graph"
        }
        
        for component_name, expected_family in component_families.items():
            component_info = self.registry.get_component_info(component_name)
            assert component_info is not None, f"Component {component_name} info not found"
            assert component_info.get('family') == expected_family, \
                f"Component {component_name} has incorrect family. Expected: {expected_family}, Got: {component_info.get('family')}"
    
    def test_component_instantiation_with_invalid_config(self):
        """Test component behavior with invalid configurations."""
        component_name = "multihead_graph_attention"
        
        # Test with missing required parameters
        invalid_config = {"dropout": 0.1}  # Missing d_model, n_heads, etc.
        
        with pytest.raises((ValueError, TypeError, KeyError)):
            self.registry.create_component(component_name, invalid_config)
    
    def test_component_forward_pass_shapes(self):
        """Test that components produce expected output shapes."""
        # Test MultiHeadGraphAttention
        config = {
            "d_model": 64,
            "n_heads": 8,
            "dropout": 0.1,
            "graph_dim": 32
        }
        
        component = self.registry.create_component("multihead_graph_attention", config)
        
        # Create test input
        batch_size, seq_len, d_model = 2, 10, 64
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Test forward pass (if component supports it)
        try:
            output = component(x)
            assert output.shape == x.shape, "Output shape should match input shape"
        except (NotImplementedError, AttributeError):
            # Some components might need additional inputs
            pass