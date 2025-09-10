"""Tests for backbone registry functionality."""

import pytest
import torch
import torch.nn as nn
from layers.modular.backbone.registry import BackboneRegistry, get_backbone_component
from layers.modular.core.registry import ComponentFamily


class TestBackboneRegistry:
    """Test suite for backbone registry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = BackboneRegistry()
        self.test_config = {
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8
        }
    
    def test_registry_initialization(self):
        """Test that registry initializes correctly."""
        assert isinstance(self.registry, BackboneRegistry)
        assert self.registry.component_family == ComponentFamily.BACKBONE
    
    def test_get_available_components(self):
        """Test getting list of available components."""
        components = self.registry.get_available_components()
        assert isinstance(components, list)
        assert len(components) > 0
        
        # Check that expected components are available
        expected_components = [
            'chronos_backbone',
            't5_backbone', 
            'bert_backbone',
            'simple_transformer_backbone'
        ]
        
        for component in expected_components:
            assert component in components
    
    def test_component_creation(self):
        """Test creating components through registry."""
        # Test each registered component
        components = self.registry.get_available_components()
        
        for component_name in components:
            try:
                component = self.registry.create_component(
                    component_name, 
                    **self.test_config
                )
                assert isinstance(component, nn.Module)
                
                # Test forward pass with dummy input
                batch_size, seq_len = 2, 100
                dummy_input = torch.randn(batch_size, seq_len, self.test_config['d_model'])
                
                with torch.no_grad():
                    output = component(dummy_input)
                    assert output is not None
                    assert output.shape[0] == batch_size
                    assert output.shape[-1] == self.test_config['d_model']
                    
            except Exception as e:
                pytest.fail(f"Failed to create or test component {component_name}: {e}")
    
    def test_factory_function(self):
        """Test the factory function."""
        component = get_backbone_component(
            'simple_transformer_backbone',
            **self.test_config
        )
        
        assert isinstance(component, nn.Module)
        
        # Test forward pass
        batch_size, seq_len = 2, 50
        dummy_input = torch.randn(batch_size, seq_len, self.test_config['d_model'])
        
        with torch.no_grad():
            output = component(dummy_input)
            assert output.shape == dummy_input.shape
    
    def test_invalid_component_name(self):
        """Test handling of invalid component names."""
        with pytest.raises((KeyError, ValueError)):
            self.registry.create_component('invalid_backbone', **self.test_config)
    
    def test_missing_required_config(self):
        """Test handling of missing required configuration."""
        with pytest.raises((TypeError, ValueError)):
            self.registry.create_component('simple_transformer_backbone')
    
    def test_component_info(self):
        """Test getting component information."""
        components = self.registry.get_available_components()
        
        for component_name in components:
            info = self.registry.get_component_info(component_name)
            assert isinstance(info, dict)
            assert 'name' in info
            assert 'component_class' in info
            assert 'component_type' in info
            assert info['component_type'] == ComponentFamily.BACKBONE


if __name__ == '__main__':
    pytest.main([__file__])