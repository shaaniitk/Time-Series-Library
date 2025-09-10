"""Integration tests for all registry components working together."""

import pytest
import torch
import torch.nn as nn
from layers.modular.core.register_components import component_registry
from layers.modular.core.registry import ComponentFamily
from layers.modular.backbone.registry import get_backbone_component
from layers.modular.feedforward.registry import get_feedforward_component
from layers.modular.output.registry import get_output_component
from layers.modular.attention.registry import get_attention_component
from layers.modular.decomposition.registry import get_decomposition_component


class TestRegistryIntegration:
    """Integration tests for the complete registry system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.d_model = 512
        self.batch_size = 2
        self.seq_len = 100
        
        self.base_config = {
            'd_model': self.d_model,
            'n_heads': 8,
            'dropout': 0.1
        }
    
    def test_component_registry_initialization(self):
        """Test that the global component registry is properly initialized."""
        assert component_registry is not None
        
        # Check that all component families are registered
        expected_families = [
            ComponentFamily.ATTENTION,
            ComponentFamily.DECOMPOSITION,
            ComponentFamily.BACKBONE,
            ComponentFamily.FEEDFORWARD,
            ComponentFamily.OUTPUT
        ]
        
        registered_families = set()
        for component_name in component_registry.get_all_components():
            info = component_registry.get_component_info(component_name)
            registered_families.add(info['component_type'])
        
        for family in expected_families:
            assert family in registered_families, f"Component family {family} not found in registry"
    
    def test_all_component_families_available(self):
        """Test that components from all families are available."""
        all_components = component_registry.get_all_components()
        assert len(all_components) > 0
        
        # Group components by family
        families = {}
        for component_name in all_components:
            info = component_registry.get_component_info(component_name)
            family = info['component_type']
            if family not in families:
                families[family] = []
            families[family].append(component_name)
        
        # Verify we have components from each expected family
        expected_families = [
            ComponentFamily.ATTENTION,
            ComponentFamily.DECOMPOSITION, 
            ComponentFamily.BACKBONE,
            ComponentFamily.FEEDFORWARD,
            ComponentFamily.OUTPUT
        ]
        
        for family in expected_families:
            assert family in families, f"No components found for family {family}"
            assert len(families[family]) > 0, f"No components registered for family {family}"
    
    def test_end_to_end_model_construction(self):
        """Test building a complete model using components from different registries."""
        # Create components from different families
        attention = get_attention_component(
            'multi_head_attention',
            d_model=self.d_model,
            n_heads=8,
            dropout=0.1
        )
        
        backbone = get_backbone_component(
            'simple_transformer_backbone',
            d_model=self.d_model,
            n_layers=2,
            n_heads=8
        )
        
        feedforward = get_feedforward_component(
            'standard_ffn',
            d_model=self.d_model,
            d_ff=2048,
            dropout=0.1
        )
        
        output_layer = get_output_component(
            'linear_output',
            d_model=self.d_model,
            output_dim=1
        )
        
        # Test that all components are created successfully
        assert isinstance(attention, nn.Module)
        assert isinstance(backbone, nn.Module)
        assert isinstance(feedforward, nn.Module)
        assert isinstance(output_layer, nn.Module)
        
        # Test forward pass through the pipeline
        dummy_input = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        with torch.no_grad():
            # Pass through backbone
            x = backbone(dummy_input)
            assert x.shape == dummy_input.shape
            
            # Pass through feedforward
            x = feedforward(x)
            assert x.shape == dummy_input.shape
            
            # Pass through output layer
            output = output_layer(x)
            assert output.shape == (self.batch_size, self.seq_len, 1)
    
    def test_component_compatibility(self):
        """Test that components from different families are compatible."""
        # Test different combinations of components
        test_combinations = [
            {
                'backbone': 'simple_transformer_backbone',
                'feedforward': 'standard_ffn',
                'output': 'linear_output'
            },
            {
                'backbone': 'chronos_backbone',
                'feedforward': 'gated_ffn',
                'output': 'forecasting_head'
            },
            {
                'backbone': 'bert_backbone',
                'feedforward': 'conv_ffn',
                'output': 'regression_head'
            }
        ]
        
        for combination in test_combinations:
            try:
                # Create backbone
                backbone_config = {
                    'd_model': self.d_model,
                    'n_layers': 2,
                    'n_heads': 8
                }
                backbone = get_backbone_component(
                    combination['backbone'],
                    **backbone_config
                )
                
                # Create feedforward
                ff_config = {
                    'd_model': self.d_model,
                    'd_ff': 2048,
                    'dropout': 0.1
                }
                if 'conv' in combination['feedforward']:
                    ff_config['kernel_size'] = 3
                    ff_config['padding'] = 1
                
                feedforward = get_feedforward_component(
                    combination['feedforward'],
                    **ff_config
                )
                
                # Create output
                output_config = {
                    'd_model': self.d_model,
                    'output_dim': 1
                }
                if 'forecasting' in combination['output']:
                    output_config['horizon'] = 12
                
                output_layer = get_output_component(
                    combination['output'],
                    **output_config
                )
                
                # Test forward pass
                dummy_input = torch.randn(self.batch_size, self.seq_len, self.d_model)
                
                with torch.no_grad():
                    x = backbone(dummy_input)
                    x = feedforward(x)
                    output = output_layer(x)
                    
                    assert output is not None
                    assert len(output.shape) >= 2
                    assert output.shape[0] == self.batch_size
                    
            except Exception as e:
                pytest.fail(f"Failed compatibility test for combination {combination}: {e}")
    
    def test_registry_component_info_consistency(self):
        """Test that component info is consistent across registries."""
        all_components = component_registry.get_all_components()
        
        for component_name in all_components:
            info = component_registry.get_component_info(component_name)
            
            # Check required fields
            required_fields = ['name', 'component_class', 'component_type']
            for field in required_fields:
                assert field in info, f"Missing field {field} in component {component_name}"
            
            # Check that component_class is a valid class
            assert isinstance(info['component_class'], type)
            assert issubclass(info['component_class'], nn.Module)
            
            # Check that component_type is a valid ComponentFamily
            assert isinstance(info['component_type'], ComponentFamily)
    
    def test_factory_functions_consistency(self):
        """Test that factory functions work consistently."""
        factory_tests = [
            {
                'factory': get_backbone_component,
                'component': 'simple_transformer_backbone',
                'config': {'d_model': 256, 'n_layers': 2, 'n_heads': 4}
            },
            {
                'factory': get_feedforward_component,
                'component': 'standard_ffn',
                'config': {'d_model': 256, 'd_ff': 1024}
            },
            {
                'factory': get_output_component,
                'component': 'linear_output',
                'config': {'d_model': 256, 'output_dim': 1}
            }
        ]
        
        for test in factory_tests:
            try:
                component = test['factory'](test['component'], **test['config'])
                assert isinstance(component, nn.Module)
                
                # Test forward pass
                dummy_input = torch.randn(2, 50, test['config']['d_model'])
                with torch.no_grad():
                    output = component(dummy_input)
                    assert output is not None
                    
            except Exception as e:
                pytest.fail(f"Factory function test failed for {test['component']}: {e}")
    
    def test_error_handling_consistency(self):
        """Test that error handling is consistent across registries."""
        factories = [
            get_backbone_component,
            get_feedforward_component,
            get_output_component
        ]
        
        for factory in factories:
            # Test invalid component name
            with pytest.raises((KeyError, ValueError)):
                factory('invalid_component_name', d_model=512)
            
            # Test missing required config (should raise TypeError or ValueError)
            with pytest.raises((TypeError, ValueError)):
                factory('standard_ffn')  # Missing required parameters


if __name__ == '__main__':
    pytest.main([__file__])