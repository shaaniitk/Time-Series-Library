"""Tests for output registry functionality."""

import pytest
import torch
import torch.nn as nn
from layers.modular.output.registry import OutputRegistry, get_output_component
from layers.modular.core.registry import ComponentFamily


class TestOutputRegistry:
    """Test suite for output registry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = OutputRegistry()
        self.test_config = {
            'd_model': 512,
            'output_dim': 1
        }
    
    def test_registry_initialization(self):
        """Test that registry initializes correctly."""
        assert isinstance(self.registry, OutputRegistry)
        assert self.registry.component_family == ComponentFamily.OUTPUT
    
    def test_get_available_components(self):
        """Test getting list of available components."""
        components = self.registry.get_available_components()
        assert isinstance(components, list)
        assert len(components) > 0
        
        # Check that expected components are available
        expected_components = [
            'linear_output',
            'forecasting_head',
            'regression_head'
        ]
        
        for component in expected_components:
            assert component in components
    
    def test_linear_output_creation(self):
        """Test creating linear output component."""
        component = self.registry.create_component(
            'linear_output',
            **self.test_config
        )
        
        assert isinstance(component, nn.Module)
        
        # Test forward pass
        batch_size, seq_len = 2, 100
        dummy_input = torch.randn(batch_size, seq_len, self.test_config['d_model'])
        
        with torch.no_grad():
            output = component(dummy_input)
            assert output.shape == (batch_size, seq_len, self.test_config['output_dim'])
    
    def test_forecasting_head_creation(self):
        """Test creating forecasting head component."""
        forecast_config = self.test_config.copy()
        forecast_config['horizon'] = 24  # 24-step ahead forecasting
        
        component = self.registry.create_component(
            'forecasting_head',
            **forecast_config
        )
        
        assert isinstance(component, nn.Module)
        
        # Test forward pass
        batch_size, seq_len = 2, 100
        dummy_input = torch.randn(batch_size, seq_len, self.test_config['d_model'])
        
        with torch.no_grad():
            output = component(dummy_input)
            # Output should be (batch_size, horizon, output_dim)
            assert output.shape == (batch_size, forecast_config['horizon'], self.test_config['output_dim'])
    
    def test_regression_head_creation(self):
        """Test creating regression head component."""
        regression_config = self.test_config.copy()
        regression_config['output_dim'] = 5  # Multi-target regression
        
        component = self.registry.create_component(
            'regression_head',
            **regression_config
        )
        
        assert isinstance(component, nn.Module)
        
        # Test forward pass
        batch_size, seq_len = 2, 100
        dummy_input = torch.randn(batch_size, seq_len, self.test_config['d_model'])
        
        with torch.no_grad():
            output = component(dummy_input)
            # Output should be (batch_size, output_dim) for regression
            assert output.shape == (batch_size, regression_config['output_dim'])
    
    def test_factory_function(self):
        """Test the factory function."""
        component = get_output_component(
            'linear_output',
            **self.test_config
        )
        
        assert isinstance(component, nn.Module)
        
        # Test forward pass
        batch_size, seq_len = 2, 50
        dummy_input = torch.randn(batch_size, seq_len, self.test_config['d_model'])
        
        with torch.no_grad():
            output = component(dummy_input)
            assert output.shape == (batch_size, seq_len, self.test_config['output_dim'])
    
    def test_invalid_component_name(self):
        """Test handling of invalid component names."""
        with pytest.raises((KeyError, ValueError)):
            self.registry.create_component('invalid_output', **self.test_config)
    
    def test_missing_required_config(self):
        """Test handling of missing required configuration."""
        with pytest.raises((TypeError, ValueError)):
            self.registry.create_component('linear_output')
    
    def test_component_info(self):
        """Test getting component information."""
        components = self.registry.get_available_components()
        
        for component_name in components:
            info = self.registry.get_component_info(component_name)
            assert isinstance(info, dict)
            assert 'name' in info
            assert 'component_class' in info
            assert 'component_type' in info
            assert info['component_type'] == ComponentFamily.OUTPUT
    
    def test_all_components_creation(self):
        """Test creating all registered components."""
        components = self.registry.get_available_components()
        
        for component_name in components:
            try:
                # Use appropriate config for each component type
                config = self.test_config.copy()
                
                if 'forecasting' in component_name:
                    config['horizon'] = 12
                elif 'regression' in component_name:
                    config['output_dim'] = 3
                
                component = self.registry.create_component(
                    component_name,
                    **config
                )
                
                assert isinstance(component, nn.Module)
                
                # Test forward pass
                batch_size, seq_len = 2, 50
                dummy_input = torch.randn(batch_size, seq_len, self.test_config['d_model'])
                
                with torch.no_grad():
                    output = component(dummy_input)
                    assert output is not None
                    assert len(output.shape) >= 2  # At least batch and feature dimensions
                    assert output.shape[0] == batch_size
                    
            except Exception as e:
                pytest.fail(f"Failed to create or test component {component_name}: {e}")
    
    def test_different_output_dimensions(self):
        """Test components with different output dimensions."""
        test_cases = [
            {'output_dim': 1},
            {'output_dim': 5},
            {'output_dim': 10}
        ]
        
        for test_case in test_cases:
            config = self.test_config.copy()
            config.update(test_case)
            
            component = self.registry.create_component(
                'linear_output',
                **config
            )
            
            batch_size, seq_len = 2, 50
            dummy_input = torch.randn(batch_size, seq_len, config['d_model'])
            
            with torch.no_grad():
                output = component(dummy_input)
                assert output.shape == (batch_size, seq_len, config['output_dim'])
    
    def test_forecasting_different_horizons(self):
        """Test forecasting head with different horizons."""
        horizons = [1, 12, 24, 48]
        
        for horizon in horizons:
            config = self.test_config.copy()
            config['horizon'] = horizon
            
            component = self.registry.create_component(
                'forecasting_head',
                **config
            )
            
            batch_size, seq_len = 2, 100
            dummy_input = torch.randn(batch_size, seq_len, config['d_model'])
            
            with torch.no_grad():
                output = component(dummy_input)
                assert output.shape == (batch_size, horizon, config['output_dim'])


if __name__ == '__main__':
    pytest.main([__file__])