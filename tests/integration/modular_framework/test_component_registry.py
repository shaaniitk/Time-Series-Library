"""
Test Suite for Component Registry and Configuration System

This file focuses specifically on testing the component registry functionality,
configuration management, and dynamic component loading.
"""

import pytest
import torch
import torch.nn as nn
import yaml
import tempfile
import os
import sys
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modular components with graceful fallback
try:
    from layers.modular.core.registry import UnifiedRegistry as ComponentRegistry, unified_registry as get_global_registry
from layers.modular.core.base_interfaces import BaseComponent, ComponentType
from layers.modular.core.config_schemas import ComponentConfig
    MODULAR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Modular components not available: {e}")
    MODULAR_AVAILABLE = False


class TestComponentRegistry:
    """Comprehensive tests for the component registry system"""
    
    def test_registry_singleton_pattern(self):
        """Test that get_global_registry returns the same instance"""
        if not MODULAR_AVAILABLE:
            pytest.skip("Modular components not available")
        
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        
        assert registry1 is registry2, "Registry should be a singleton"
    
    def test_component_metadata_storage(self):
        """Test that component metadata is properly stored and retrieved"""
        registry = ComponentRegistry()
        
        class TestComponent(BaseComponent):
            def __init__(self, config):
                super().__init__(config)
            
            def forward(self, x):
                return x
            
            def get_config(self):
                return {}
        
        metadata = {
            'description': 'Test component for unit testing',
            'version': '1.0.0',
            'author': 'Test Suite',
            'performance_characteristics': {
                'memory_usage': 'low',
                'compute_complexity': 'O(n)',
                'supports_batching': True
            },
            'requirements': ['torch>=1.8.0'],
            'parameters': {
                'required': ['d_model'],
                'optional': ['dropout', 'activation']
            }
        }
        
        registry.register(
            component_type='attention',
            component_name='test_component',
            component_class=TestComponent,
            metadata=metadata
        )
        
        # Retrieve metadata
        stored_metadata = registry._metadata['attention']['test_component']
        
        # Check that all original metadata is preserved
        assert stored_metadata['description'] == metadata['description']
        assert stored_metadata['version'] == metadata['version']
        assert stored_metadata['author'] == metadata['author']
        assert stored_metadata['performance_characteristics'] == metadata['performance_characteristics']
        assert stored_metadata['requirements'] == metadata['requirements']
        assert stored_metadata['parameters'] == metadata['parameters']
        
        # Check that registry adds its own metadata
        assert 'class_name' in stored_metadata
        assert 'module' in stored_metadata
        assert 'registered_name' in stored_metadata
        assert stored_metadata['registered_name'] == 'test_component'
    
    def test_component_overwrite_warning(self):
        """Test that overwriting components generates appropriate warnings"""
        registry = ComponentRegistry()
        
        class ComponentV1(BaseComponent):
            def forward(self, x): return x
            def get_config(self): return {'version': 1}
        
        class ComponentV2(BaseComponent):
            def forward(self, x): return x * 2
            def get_config(self): return {'version': 2}
        
        # Register first version
        registry.register('processor', 'test_overwrite', ComponentV1)
        
        # Register second version (should warn)
        with patch('layers.modular.core.registry.logger') as mock_logger:
            registry.register('processor', 'test_overwrite', ComponentV2)
            mock_logger.warning.assert_called_once()
            
            # Verify the warning message mentions overwriting
            warning_message = mock_logger.warning.call_args[0][0]
            assert 'Overwriting' in warning_message
            assert 'test_overwrite' in warning_message
        
        # Verify that the new component is actually registered
        retrieved_class = registry.get('processor', 'test_overwrite')
        instance = retrieved_class({})
        assert instance.get_config()['version'] == 2
    
    def test_component_listing_and_discovery(self):
        """Test component listing and discovery functionality"""
        registry = ComponentRegistry()
        
        # Register multiple components
        components_to_register = [
            ('attention', 'component_a', Mock),
            ('attention', 'component_b', Mock),
            ('processor', 'component_c', Mock),
            ('loss', 'component_d', Mock),
        ]
        
        for comp_type, comp_name, comp_class in components_to_register:
            registry.register(comp_type, comp_name, comp_class)
        
        # Test listing by category
        attention_components = list(registry._components['attention'].keys())
        assert 'component_a' in attention_components
        assert 'component_b' in attention_components
        assert len(attention_components) >= 2
        
        # Test that components are in correct categories
        assert 'component_c' in registry._components['processor']
        assert 'component_d' in registry._components['loss']
        assert 'component_c' not in registry._components['attention']
    
    def test_component_creation_with_different_config_types(self):
        """Test component creation with various configuration formats"""
        registry = ComponentRegistry()
        
        class FlexibleComponent(BaseComponent):
            def __init__(self, config):
                super().__init__(config)
                # Handle different config types
                if isinstance(config, dict):
                    self.param1 = config.get('param1', 'default1')
                    self.param2 = config.get('param2', 42)
                else:
                    # Handle object-like configs
                    self.param1 = getattr(config, 'param1', 'default1')
                    self.param2 = getattr(config, 'param2', 42)
            
            def forward(self, x):
                return x
            
            def get_config(self):
                return {'param1': self.param1, 'param2': self.param2}
        
        registry.register('embedding', 'flexible_component', FlexibleComponent)
        
        # Test with dictionary config
        component1 = registry.create('embedding', 'flexible_component', {
            'param1': 'custom_value',
            'param2': 100
        })
        assert component1.param1 == 'custom_value'
        assert component1.param2 == 100
        
        # Test with object config
        from types import SimpleNamespace
        config_obj = SimpleNamespace()
        config_obj.param1 = 'object_value'
        config_obj.param2 = 200
        
        component2 = registry.create('embedding', 'flexible_component', config_obj)
        assert component2.param1 == 'object_value'
        assert component2.param2 == 200
        
        # Test with partial config (should use defaults)
        component3 = registry.create('embedding', 'flexible_component', {'param1': 'partial'})
        assert component3.param1 == 'partial'
        assert component3.param2 == 42  # default value


class TestConfigurationManagement:
    """Test configuration loading, validation, and management"""
    
    def test_yaml_config_comprehensive(self):
        """Test comprehensive YAML configuration handling"""
        
        # Complex configuration with multiple scenarios
        complex_config_yaml = """
model_type: "ModularAutoformer"
version: "2.0"

# Global model parameters
global_params:
  seq_len: 96
  pred_len: 24
  d_model: 512
  batch_size: 32

# Component specifications
components:
  decomposition:
    type: "LearnableSeriesDecomp"
    params:
      d_model: ${global_params.d_model}
      kernel_size: 25
      adaptive_kernel: true
      feature_specific: true
    metadata:
      memory_efficient: true
      computational_cost: "medium"
  
  attention:
    type: "EnhancedAutoCorrelation"
    params:
      d_model: ${global_params.d_model}
      num_heads: 8
      factor: 1
      attention_dropout: 0.1
      use_mixed_precision: true
    alternatives:
      - type: "StandardAutoCorrelation"
        conditions: ["memory_limited"]
      - type: "OptimizedAutoCorrelation"
        conditions: ["large_sequences"]
  
  encoder:
    type: "EnhancedEncoder"
    params:
      layers: 2
      d_model: ${global_params.d_model}
      d_ff: 2048
      dropout: 0.1
      activation: "gelu"
      gated_ffn: true
    layer_configs:
      - layer_id: 0
        special_params:
          attention_scale: 1.0
      - layer_id: 1
        special_params:
          attention_scale: 0.8
  
  decoder:
    type: "EnhancedDecoder"
    params:
      layers: 1
      d_model: ${global_params.d_model}
      c_out: 7
    
  sampling:
    type: "BayesianSampling"
    params:
      num_samples: 50
      output_dim: 7
      uncertainty_method: "monte_carlo"
      temperature: 1.0

# Training configuration
training:
  optimizer:
    type: "AdamW"
    params:
      lr: 0.0001
      weight_decay: 0.01
      betas: [0.9, 0.999]
  
  scheduler:
    type: "CosineAnnealingLR"
    params:
      T_max: 100
      eta_min: 0.00001
  
  loss:
    type: "bayesian_mse"
    params:
      kl_weight: 0.00001
      uncertainty_weight: 0.1
      reduction: "mean"
  
  regularization:
    gradient_clipping:
      enabled: true
      max_norm: 1.0
    weight_decay: 0.01
    dropout_schedule:
      initial: 0.1
      final: 0.05
      decay_steps: 50

# Evaluation configuration
evaluation:
  metrics: ["mse", "mae", "mape", "uncertainty_score"]
  uncertainty_quantiles: [0.1, 0.25, 0.5, 0.75, 0.9]
  
# Environment and deployment
environment:
  device: "auto"  # auto-detect GPU/CPU
  mixed_precision: true
  compile_model: false
  
deployment:
  export_format: ["onnx", "torchscript"]
  optimization_level: "O2"
"""
        
        # Parse complex configuration
        config = yaml.safe_load(complex_config_yaml)
        
        # Test top-level structure
        assert config['model_type'] == 'ModularAutoformer'
        assert config['version'] == '2.0'
        assert 'global_params' in config
        assert 'components' in config
        assert 'training' in config
        assert 'evaluation' in config
        assert 'environment' in config
        assert 'deployment' in config
        
        # Test global parameters
        global_params = config['global_params']
        assert global_params['seq_len'] == 96
        assert global_params['pred_len'] == 24
        assert global_params['d_model'] == 512
        assert global_params['batch_size'] == 32
        
        # Test component configurations
        components = config['components']
        
        # Test decomposition component
        decomp = components['decomposition']
        assert decomp['type'] == 'LearnableSeriesDecomp'
        assert decomp['params']['kernel_size'] == 25
        assert decomp['params']['adaptive_kernel'] is True
        assert decomp['metadata']['memory_efficient'] is True
        
        # Test attention component with alternatives
        attention = components['attention']
        assert attention['type'] == 'EnhancedAutoCorrelation'
        assert 'alternatives' in attention
        assert len(attention['alternatives']) == 2
        
        # Test encoder with layer-specific configs
        encoder = components['encoder']
        assert encoder['type'] == 'EnhancedEncoder'
        assert 'layer_configs' in encoder
        assert len(encoder['layer_configs']) == 2
        
        # Test training configuration
        training = config['training']
        assert training['optimizer']['type'] == 'AdamW'
        assert training['loss']['type'] == 'bayesian_mse'
        assert training['regularization']['gradient_clipping']['enabled'] is True
    
    def test_config_validation_comprehensive(self):
        """Test comprehensive configuration validation"""
        
        def validate_config_comprehensive(config):
            """Comprehensive configuration validator"""
            errors = []
            warnings = []
            
            # Required top-level fields
            required_fields = ['model_type', 'components']
            for field in required_fields:
                if field not in config:
                    errors.append(f"Missing required field: {field}")
            
            # Validate model_type
            if 'model_type' in config:
                valid_types = ['ModularAutoformer', 'HFAutoformer', 'EnhancedAutoformer']
                if config['model_type'] not in valid_types:
                    warnings.append(f"Unrecognized model_type: {config['model_type']}")
            
            # Validate components
            if 'components' in config:
                required_components = ['decomposition', 'attention', 'encoder', 'decoder', 'sampling']
                components = config['components']
                
                for component in required_components:
                    if component not in components:
                        errors.append(f"Missing required component: {component}")
                    else:
                        comp_config = components[component]
                        
                        # Validate component structure
                        if 'type' not in comp_config:
                            errors.append(f"Component '{component}' missing type specification")
                        
                        # Validate parameters if present
                        if 'params' in comp_config:
                            params = comp_config['params']
                            
                            # Component-specific validation
                            if component == 'attention' and 'd_model' in params and 'num_heads' in params:
                                if params['d_model'] % params['num_heads'] != 0:
                                    errors.append(f"d_model ({params['d_model']}) must be divisible by num_heads ({params['num_heads']})")
                            
                            if component == 'encoder' and 'layers' in params:
                                if params['layers'] < 1:
                                    errors.append(f"Encoder must have at least 1 layer, got {params['layers']}")
                            
                            if component == 'sampling' and params.get('type') == 'bayesian' and 'num_samples' in params:
                                if params['num_samples'] < 1:
                                    errors.append(f"Bayesian sampling requires at least 1 sample, got {params['num_samples']}")
            
            # Validate training configuration if present
            if 'training' in config:
                training = config['training']
                
                if 'optimizer' in training:
                    opt = training['optimizer']
                    if 'type' not in opt:
                        warnings.append("Optimizer type not specified")
                    
                    if 'params' in opt and 'lr' in opt['params']:
                        lr = opt['params']['lr']
                        if lr <= 0 or lr > 1:
                            warnings.append(f"Learning rate {lr} seems unusual (should be between 0 and 1)")
                
                if 'loss' in training:
                    loss = training['loss']
                    if loss.get('type') == 'bayesian_mse' and 'params' in loss:
                        kl_weight = loss['params'].get('kl_weight', 0)
                        if kl_weight < 0:
                            errors.append("KL weight cannot be negative")
                        elif kl_weight > 0.1:
                            warnings.append(f"KL weight {kl_weight} is very high, might dominate training")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }
        
        # Test valid configuration
        valid_config = {
            'model_type': 'ModularAutoformer',
            'components': {
                'decomposition': {'type': 'SeriesDecomposition', 'params': {'kernel_size': 25}},
                'attention': {'type': 'AutoCorrelation', 'params': {'d_model': 512, 'num_heads': 8}},
                'encoder': {'type': 'StandardEncoder', 'params': {'layers': 2}},
                'decoder': {'type': 'StandardDecoder', 'params': {'layers': 1}},
                'sampling': {'type': 'DeterministicSampling', 'params': {}}
            },
            'training': {
                'optimizer': {'type': 'Adam', 'params': {'lr': 0.001}},
                'loss': {'type': 'mse', 'params': {}}
            }
        }
        
        result = validate_config_comprehensive(valid_config)
        assert result['valid'], f"Valid config failed: {result['errors']}"
        assert len(result['errors']) == 0
        
        # Test invalid configurations
        invalid_configs = [
            # Missing model_type
            {'components': valid_config['components']},
            
            # Missing required component
            {
                'model_type': 'ModularAutoformer',
                'components': {k: v for k, v in valid_config['components'].items() if k != 'attention'}
            },
            
            # Invalid d_model/num_heads ratio
            {
                'model_type': 'ModularAutoformer',
                'components': {
                    **valid_config['components'],
                    'attention': {'type': 'AutoCorrelation', 'params': {'d_model': 511, 'num_heads': 8}}
                }
            },
            
            # Invalid encoder layers
            {
                'model_type': 'ModularAutoformer',
                'components': {
                    **valid_config['components'],
                    'encoder': {'type': 'StandardEncoder', 'params': {'layers': 0}}
                }
            }
        ]
        
        for i, invalid_config in enumerate(invalid_configs):
            result = validate_config_comprehensive(invalid_config)
            assert not result['valid'], f"Invalid config {i} passed validation: {invalid_config}"
            assert len(result['errors']) > 0
    
    def test_config_templating_and_inheritance(self):
        """Test configuration templating and inheritance"""
        
        # Base template
        base_template = {
            'model_type': 'ModularAutoformer',
            'global_params': {
                'd_model': 512,
                'dropout': 0.1,
                'activation': 'gelu'
            },
            'components': {
                'decomposition': {
                    'type': 'SeriesDecomposition',
                    'params': {'kernel_size': 25}
                },
                'attention': {
                    'type': 'StandardAutoCorrelation',
                    'params': {
                        'd_model': '${global_params.d_model}',
                        'num_heads': 8,
                        'dropout': '${global_params.dropout}'
                    }
                },
                'encoder': {
                    'type': 'StandardEncoder',
                    'params': {
                        'layers': 2,
                        'd_model': '${global_params.d_model}',
                        'activation': '${global_params.activation}'
                    }
                }
            }
        }
        
        # Function to resolve template variables (simplified)
        def resolve_template_vars(config):
            """Simplified template variable resolution"""
            import json
            import re
            
            config_str = json.dumps(config)
            
            # Find all template variables
            var_pattern = r'\$\{([^}]+)\}'
            
            def replace_var(match):
                var_path = match.group(1)
                path_parts = var_path.split('.')
                
                value = config
                for part in path_parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return match.group(0)  # Return original if not found
                
                return str(value)
            
            resolved_str = re.sub(var_pattern, replace_var, config_str)
            return json.loads(resolved_str)
        
        # Test template resolution
        resolved_config = resolve_template_vars(base_template)
        
        # Verify that template variables were resolved
        attention_params = resolved_config['components']['attention']['params']
        assert attention_params['d_model'] == 512  # Should be resolved from global_params
        assert attention_params['dropout'] == 0.1
        
        encoder_params = resolved_config['components']['encoder']['params']
        assert encoder_params['d_model'] == 512
        assert encoder_params['activation'] == 'gelu'
        
        # Test configuration inheritance/extension
        def extend_config(base_config, extensions):
            """Deep merge configuration extensions"""
            import copy
            
            result = copy.deepcopy(base_config)
            
            def deep_merge(target, source):
                for key, value in source.items():
                    if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                        deep_merge(target[key], value)
                    else:
                        target[key] = value
            
            deep_merge(result, extensions)
            return result
        
        # Test extending base configuration for Bayesian variant
        bayesian_extensions = {
            'components': {
                'attention': {
                    'type': 'BayesianAutoCorrelation',
                    'params': {
                        'uncertainty_samples': 50
                    }
                },
                'sampling': {
                    'type': 'BayesianSampling',
                    'params': {
                        'num_samples': 100,
                        'kl_weight': 0.00001
                    }
                }
            },
            'training': {
                'loss': {
                    'type': 'bayesian_mse',
                    'params': {'kl_weight': 0.00001}
                }
            }
        }
        
        bayesian_config = extend_config(base_template, bayesian_extensions)
        
        # Verify extension worked correctly
        assert bayesian_config['components']['attention']['type'] == 'BayesianAutoCorrelation'
        assert bayesian_config['components']['attention']['params']['uncertainty_samples'] == 50
        assert bayesian_config['components']['attention']['params']['d_model'] == '${global_params.d_model}'  # Original preserved
        
        assert 'sampling' in bayesian_config['components']
        assert bayesian_config['components']['sampling']['type'] == 'BayesianSampling'
        
        assert 'training' in bayesian_config
        assert bayesian_config['training']['loss']['type'] == 'bayesian_mse'


class TestDynamicComponentLoading:
    """Test dynamic component loading and runtime component swapping"""
    
    def test_runtime_component_swapping(self):
        """Test that components can be swapped at runtime"""
        
        # Mock model that supports component swapping
        class SwappableModel(nn.Module):
            def __init__(self, initial_components):
                super().__init__()
                self.components = initial_components
            
            def swap_component(self, component_type, new_component):
                """Swap a component at runtime"""
                old_component = self.components.get(component_type)
                self.components[component_type] = new_component
                return old_component
            
            def forward(self, x):
                # Use current components
                for component_name, component in self.components.items():
                    if hasattr(component, 'forward'):
                        x = component.forward(x)
                return x
        
        # Create initial components
        class ComponentA(nn.Module):
            def forward(self, x): return x * 2
        
        class ComponentB(nn.Module):
            def forward(self, x): return x * 3
        
        class ComponentC(nn.Module):
            def forward(self, x): return x + 1
        
        initial_components = {
            'processor': ComponentA(),
            'transformer': ComponentB()
        }
        
        model = SwappableModel(initial_components)
        
        # Test initial behavior
        test_input = torch.tensor([1.0, 2.0, 3.0])
        initial_output = model(test_input)
        expected_initial = test_input * 2 * 3  # ComponentA then ComponentB
        assert torch.allclose(initial_output, expected_initial)
        
        # Swap processor component
        old_processor = model.swap_component('processor', ComponentC())
        
        # Test behavior after swapping
        swapped_output = model(test_input)
        expected_swapped = (test_input + 1) * 3  # ComponentC then ComponentB
        assert torch.allclose(swapped_output, expected_swapped)
        
        # Verify old component was returned
        assert isinstance(old_processor, ComponentA)
    
    def test_lazy_component_loading(self):
        """Test lazy loading of components for memory efficiency"""
        
        # Component registry with lazy loading
        class LazyComponentRegistry:
            def __init__(self):
                self._component_factories = {}
                self._loaded_components = {}
            
            def register_factory(self, name, factory_func):
                """Register a component factory function"""
                self._component_factories[name] = factory_func
            
            def get_component(self, name, config=None):
                """Get component, loading lazily if needed"""
                if name not in self._loaded_components:
                    if name not in self._component_factories:
                        raise ValueError(f"Component '{name}' not registered")
                    
                    factory = self._component_factories[name]
                    self._loaded_components[name] = factory(config or {})
                
                return self._loaded_components[name]
            
            def unload_component(self, name):
                """Unload component to free memory"""
                if name in self._loaded_components:
                    del self._loaded_components[name]
        
        # Test lazy loading
        registry = LazyComponentRegistry()
        
        # Component creation counter for testing
        creation_count = {'count': 0}
        
        def expensive_component_factory(config):
            creation_count['count'] += 1
            
            class ExpensiveComponent(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Simulate expensive initialization
                    self.large_param = nn.Parameter(torch.randn(1000, 1000))
                
                def forward(self, x):
                    return x
            
            return ExpensiveComponent()
        
        # Register factory
        registry.register_factory('expensive_component', expensive_component_factory)
        
        # Verify component hasn't been created yet
        assert creation_count['count'] == 0
        
        # Get component (should trigger creation)
        component1 = registry.get_component('expensive_component')
        assert creation_count['count'] == 1
        
        # Get same component again (should reuse existing)
        component2 = registry.get_component('expensive_component')
        assert creation_count['count'] == 1  # Should not increment
        assert component1 is component2  # Should be same instance
        
        # Unload and reload
        registry.unload_component('expensive_component')
        component3 = registry.get_component('expensive_component')
        assert creation_count['count'] == 2  # Should increment again
        assert component3 is not component1  # Should be new instance
    
    def test_component_dependency_resolution(self):
        """Test automatic resolution of component dependencies"""
        
        class DependencyResolver:
            def __init__(self):
                self.components = {}
                self.dependencies = {}
            
            def register_component(self, name, component_class, dependencies=None):
                """Register component with optional dependencies"""
                self.components[name] = component_class
                self.dependencies[name] = dependencies or []
            
            def resolve_dependencies(self, component_name, config=None, resolved=None):
                """Recursively resolve component dependencies"""
                if resolved is None:
                    resolved = {}
                
                if component_name in resolved:
                    return resolved[component_name]
                
                if component_name not in self.components:
                    raise ValueError(f"Component '{component_name}' not found")
                
                # Resolve dependencies first
                dep_instances = {}
                for dep_name in self.dependencies[component_name]:
                    dep_instances[dep_name] = self.resolve_dependencies(dep_name, config, resolved)
                
                # Create component instance
                component_class = self.components[component_name]
                component_config = config.get(component_name, {}) if config else {}
                component_config['dependencies'] = dep_instances
                
                instance = component_class(component_config)
                resolved[component_name] = instance
                
                return instance
        
        # Test components with dependencies
        class BaseProcessor(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
            
            def forward(self, x):
                return x
        
        class NormalizationLayer(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.norm = nn.LayerNorm(config.get('d_model', 512))
            
            def forward(self, x):
                return self.norm(x)
        
        class AttentionLayer(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.attention = nn.MultiheadAttention(
                    config.get('d_model', 512),
                    config.get('num_heads', 8)
                )
                
                # Use dependency
                dependencies = config.get('dependencies', {})
                self.norm = dependencies.get('normalization')
            
            def forward(self, x):
                if self.norm:
                    x = self.norm(x)
                
                # Simplified attention
                attn_out, _ = self.attention(x, x, x)
                return attn_out
        
        class ComplexProcessor(nn.Module):
            def __init__(self, config):
                super().__init__()
                dependencies = config.get('dependencies', {})
                self.base_processor = dependencies.get('base_processor')
                self.attention = dependencies.get('attention')
            
            def forward(self, x):
                if self.base_processor:
                    x = self.base_processor(x)
                if self.attention:
                    x = self.attention(x)
                return x
        
        # Set up dependency resolver
        resolver = DependencyResolver()
        resolver.register_component('base_processor', BaseProcessor)
        resolver.register_component('normalization', NormalizationLayer)
        resolver.register_component('attention', AttentionLayer, ['normalization'])
        resolver.register_component('complex_processor', ComplexProcessor, ['base_processor', 'attention'])
        
        # Test dependency resolution
        config = {
            'base_processor': {'some_param': 'value'},
            'normalization': {'d_model': 512},
            'attention': {'d_model': 512, 'num_heads': 8},
            'complex_processor': {}
        }
        
        complex_proc = resolver.resolve_dependencies('complex_processor', config)
        
        # Verify dependencies were resolved
        assert complex_proc.base_processor is not None
        assert complex_proc.attention is not None
        assert complex_proc.attention.norm is not None
        
        # Test that the resolved component works
        test_input = torch.randn(10, 32, 512)  # [seq_len, batch, d_model]
        output = complex_proc(test_input)
        assert output.shape == test_input.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
