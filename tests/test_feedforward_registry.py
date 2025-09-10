"""Tests for feedforward registry functionality."""

import pytest
import torch
import torch.nn as nn
from layers.modular.feedforward.registry import FeedforwardRegistry, get_feedforward_component
from layers.modular.core.registry import ComponentFamily


class TestFeedforwardRegistry:
    """Test suite for feedforward registry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = FeedforwardRegistry()
        self.test_config = {
            'd_model': 512,
            'd_ff': 2048,
            'dropout': 0.1
        }
    
    def test_registry_initialization(self):
        """Test that registry initializes correctly."""
        assert isinstance(self.registry, FeedforwardRegistry)
        assert self.registry.component_family == ComponentFamily.FEEDFORWARD
    
    def test_get_available_components(self):
        """Test getting list of available components."""
        components = self.registry.get_available_components()
        assert isinstance(components, list)
        assert len(components) > 0
        
        # Check that expected components are available
        expected_components = [
            'standard_ffn',
            'gated_ffn',
            'moe_ffn',
            'conv_ffn'
        ]
        
        for component in expected_components:
            assert component in components
    
    def test_standard_ffn_creation(self):
        """Test creating standard FFN component."""
        component = self.registry.create_component(
            'standard_ffn',
            **self.test_config
        )
        
        assert isinstance(component, nn.Module)
        
        # Test forward pass
        batch_size, seq_len = 2, 100
        dummy_input = torch.randn(batch_size, seq_len, self.test_config['d_model'])
        
        with torch.no_grad():
            output = component(dummy_input)
            assert output.shape == dummy_input.shape
    
    def test_gated_ffn_creation(self):
        """Test creating gated FFN component."""
        component = self.registry.create_component(
            'gated_ffn',
            **self.test_config
        )
        
        assert isinstance(component, nn.Module)
        
        # Test forward pass
        batch_size, seq_len = 2, 100
        dummy_input = torch.randn(batch_size, seq_len, self.test_config['d_model'])
        
        with torch.no_grad():
            output = component(dummy_input)
            assert output.shape == dummy_input.shape
    
    def test_moe_ffn_creation(self):
        """Test creating MoE FFN component."""
        moe_config = self.test_config.copy()
        moe_config['num_experts'] = 4
        moe_config['top_k'] = 2
        
        component = self.registry.create_component(
            'moe_ffn',
            **moe_config
        )
        
        assert isinstance(component, nn.Module)
        
        # Test forward pass
        batch_size, seq_len = 2, 100
        dummy_input = torch.randn(batch_size, seq_len, self.test_config['d_model'])
        
        with torch.no_grad():
            output = component(dummy_input)
            assert output.shape == dummy_input.shape
    
    def test_conv_ffn_creation(self):
        """Test creating convolutional FFN component."""
        conv_config = self.test_config.copy()
        conv_config['kernel_size'] = 3
        conv_config['padding'] = 1
        
        component = self.registry.create_component(
            'conv_ffn',
            **conv_config
        )
        
        assert isinstance(component, nn.Module)
        
        # Test forward pass
        batch_size, seq_len = 2, 100
        dummy_input = torch.randn(batch_size, seq_len, self.test_config['d_model'])
        
        with torch.no_grad():
            output = component(dummy_input)
            assert output.shape == dummy_input.shape
    
    def test_factory_function(self):
        """Test the factory function."""
        component = get_feedforward_component(
            'standard_ffn',
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
            self.registry.create_component('invalid_ffn', **self.test_config)
    
    def test_missing_required_config(self):
        """Test handling of missing required configuration."""
        with pytest.raises((TypeError, ValueError)):
            self.registry.create_component('standard_ffn')
    
    def test_component_info(self):
        """Test getting component information."""
        components = self.registry.get_available_components()
        
        for component_name in components:
            info = self.registry.get_component_info(component_name)
            assert isinstance(info, dict)
            assert 'name' in info
            assert 'component_class' in info
            assert 'component_type' in info
            assert info['component_type'] == ComponentFamily.FEEDFORWARD
    
    def test_all_components_creation(self):
        """Test creating all registered components."""
        components = self.registry.get_available_components()
        
        for component_name in components:
            try:
                # Use appropriate config for each component type
                config = self.test_config.copy()
                
                if 'moe' in component_name:
                    config['num_experts'] = 4
                    config['top_k'] = 2
                elif 'conv' in component_name:
                    config['kernel_size'] = 3
                    config['padding'] = 1
                
                component = self.registry.create_component(
                    component_name,
                    **config
                )
                
                assert isinstance(component, nn.Module)
                
                # Test forward pass
                batch_size, seq_len = 2, 50
                dummy_input = torch.randn(batch_size, seq_len, config['d_model'])
                
                with torch.no_grad():
                    output = component(dummy_input)
                    assert output.shape == dummy_input.shape
                    
            except Exception as e:
                pytest.fail(f"Failed to create or test component {component_name}: {e}")


if __name__ == '__main__':
    pytest.main([__file__])