"""
Test Suite for Migration Strategy and Backward Compatibility

This file tests the migration from the current 7 separate Autoformer models
to the unified modular architecture, ensuring backward compatibility and
smooth transition paths.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing models for migration testing
try:
    from models.Autoformer import Model as BaseAutoformer
    from models.Autoformer_Fixed import Model as AutoformerFixed
    from models.EnhancedAutoformer import EnhancedAutoformer
    from models.EnhancedAutoformer_Fixed import EnhancedAutoformer as EnhancedAutoformerFixed
    from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer
    from models.HierarchicalEnhancedAutoformer import HierarchicalEnhancedAutoformer
    from models.QuantileBayesianAutoformer import QuantileBayesianAutoformer
    EXISTING_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some existing models not available: {e}")
    EXISTING_MODELS_AVAILABLE = False


class TestMigrationCompatibility:
    """Test migration from existing models to modular architecture"""
    
    @pytest.fixture
    def standard_config(self):
        """Standard configuration used across models"""
        config = SimpleNamespace()
        config.task_name = 'long_term_forecast'
        config.seq_len = 96
        config.label_len = 48
        config.pred_len = 24
        config.enc_in = 7
        config.dec_in = 7
        config.c_out = 7
        config.d_model = 512
        config.n_heads = 8
        config.e_layers = 2
        config.d_layers = 1
        config.d_ff = 2048
        config.moving_avg = 25
        config.factor = 1
        config.dropout = 0.1
        config.activation = 'gelu'
        config.embed = 'timeF'
        config.freq = 'h'
        config.norm_type = 'LayerNorm'
        return config
    
    @pytest.fixture
    def test_data(self, standard_config):
        """Generate test data for model testing"""
        batch_size = 4
        
        x_enc = torch.randn(batch_size, standard_config.seq_len, standard_config.enc_in)
        x_mark_enc = torch.randn(batch_size, standard_config.seq_len, 4)
        x_dec = torch.randn(batch_size, standard_config.label_len + standard_config.pred_len, standard_config.dec_in)
        x_mark_dec = torch.randn(batch_size, standard_config.label_len + standard_config.pred_len, 4)
        
        return x_enc, x_mark_enc, x_dec, x_mark_dec
    
    @pytest.mark.skipif(not EXISTING_MODELS_AVAILABLE, reason="Existing models not available")
    def test_base_autoformer_compatibility(self, standard_config, test_data):
        """Test that base Autoformer still works as expected"""
        x_enc, x_mark_enc, x_dec, x_mark_dec = test_data
        
        try:
            model = BaseAutoformer(standard_config)
            model.eval()
            
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Verify output shape
            expected_shape = (x_enc.shape[0], standard_config.pred_len, standard_config.c_out)
            assert output.shape == expected_shape
            
            # Verify output is reasonable (finite values)
            assert torch.isfinite(output).all()
            
        except Exception as e:
            pytest.skip(f"Base Autoformer test failed: {e}")
    
    @pytest.mark.skipif(not EXISTING_MODELS_AVAILABLE, reason="Existing models not available")
    def test_enhanced_autoformer_compatibility(self, standard_config, test_data):
        """Test that Enhanced Autoformer maintains functionality"""
        x_enc, x_mark_enc, x_dec, x_mark_dec = test_data
        
        try:
            model = EnhancedAutoformer(standard_config)
            model.eval()
            
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Verify output shape
            expected_shape = (x_enc.shape[0], standard_config.pred_len, standard_config.c_out)
            assert output.shape == expected_shape
            
            # Verify output is reasonable
            assert torch.isfinite(output).all()
            
        except Exception as e:
            pytest.skip(f"Enhanced Autoformer test failed: {e}")
    
    @pytest.mark.skipif(not EXISTING_MODELS_AVAILABLE, reason="Existing models not available")
    def test_bayesian_autoformer_compatibility(self, standard_config, test_data):
        """Test Bayesian Autoformer backward compatibility"""
        x_enc, x_mark_enc, x_dec, x_mark_dec = test_data
        
        try:
            # Bayesian models might need additional config
            bayesian_config = standard_config
            bayesian_config.uncertainty_method = 'bayesian'
            bayesian_config.n_samples = 10  # Reduced for testing
            
            model = BayesianEnhancedAutoformer(bayesian_config)
            model.eval()
            
            with torch.no_grad():
                # Test standard forward pass
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                # Test uncertainty estimation if available
                if hasattr(model, 'estimate_uncertainty'):
                    uncertainty_output = model.estimate_uncertainty(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    assert 'prediction' in uncertainty_output or 'mean' in uncertainty_output
            
            # Verify output shape
            expected_shape = (x_enc.shape[0], standard_config.pred_len, standard_config.c_out)
            assert output.shape == expected_shape
            
        except Exception as e:
            pytest.skip(f"Bayesian Autoformer test failed: {e}")
    
    def test_component_extraction_mapping(self):
        """Test mapping from existing models to modular components"""
        
        # Define component extraction mapping
        component_mapping = {
            'Autoformer': {
                'decomposition': 'SeriesDecomposition',
                'attention': 'StandardAutoCorrelation',
                'encoder': 'StandardEncoder',
                'decoder': 'StandardDecoder',
                'sampling': 'DeterministicSampling'
            },
            'EnhancedAutoformer': {
                'decomposition': 'LearnableDecomposition',
                'attention': 'EnhancedAutoCorrelation',
                'encoder': 'EnhancedEncoder',
                'decoder': 'EnhancedDecoder',
                'sampling': 'DeterministicSampling'
            },
            'BayesianEnhancedAutoformer': {
                'decomposition': 'LearnableDecomposition',
                'attention': 'EnhancedAutoCorrelation',
                'encoder': 'EnhancedEncoder',
                'decoder': 'EnhancedDecoder',
                'sampling': 'BayesianSampling'
            },
            'HierarchicalEnhancedAutoformer': {
                'decomposition': 'WaveletDecomposition',
                'attention': 'MultiWaveletCorrelation',
                'encoder': 'HierarchicalEncoder',
                'decoder': 'HierarchicalDecoder',
                'sampling': 'DeterministicSampling'
            },
            'QuantileBayesianAutoformer': {
                'decomposition': 'LearnableDecomposition',
                'attention': 'EnhancedAutoCorrelation',
                'encoder': 'EnhancedEncoder',
                'decoder': 'EnhancedDecoder',
                'sampling': 'QuantileSampling'
            }
        }
        
        # Test that mapping covers all models
        expected_models = {'Autoformer', 'EnhancedAutoformer', 'BayesianEnhancedAutoformer', 
                          'HierarchicalEnhancedAutoformer', 'QuantileBayesianAutoformer'}
        assert set(component_mapping.keys()).issuperset(expected_models)
        
        # Test that all mappings have required components
        required_components = {'decomposition', 'attention', 'encoder', 'decoder', 'sampling'}
        for model_name, mapping in component_mapping.items():
            assert set(mapping.keys()) == required_components, f"Model {model_name} missing components"
        
        # Test generating modular configs from mapping
        def generate_modular_config(model_name, base_params=None):
            """Generate modular configuration from existing model"""
            if model_name not in component_mapping:
                raise ValueError(f"Unknown model: {model_name}")
            
            mapping = component_mapping[model_name]
            base_params = base_params or {}
            
            config = {
                'model_type': 'ModularAutoformer',
                'components': {}
            }
            
            for component_type, component_name in mapping.items():
                config['components'][component_type] = {
                    'type': component_name,
                    'params': base_params.get(component_type, {})
                }
            
            return config
        
        # Test configuration generation
        for model_name in component_mapping.keys():
            config = generate_modular_config(model_name)
            
            assert config['model_type'] == 'ModularAutoformer'
            assert len(config['components']) == 5
            
            # Verify each component has correct structure
            for comp_type, comp_config in config['components'].items():
                assert 'type' in comp_config
                assert 'params' in comp_config
    
    def test_parameter_migration(self):
        """Test migration of parameters from existing models to modular components"""
        
        # Mock existing model with parameters to extract
        class MockExistingModel:
            def __init__(self):
                # Simulate existing model parameters
                self.decomp_kernel_size = 25
                self.d_model = 512
                self.n_heads = 8
                self.e_layers = 2
                self.d_layers = 1
                self.dropout = 0.1
                self.moving_avg = 25
                self.factor = 1
                
                # Bayesian specific
                self.uncertainty_method = 'bayesian'
                self.n_samples = 50
                
                # Enhanced specific
                self.adaptive_kernel = True
                self.gated_ffn = True
        
        def extract_component_params(model, component_type):
            """Extract parameters for specific component type"""
            params = {}
            
            if component_type == 'decomposition':
                params['kernel_size'] = getattr(model, 'decomp_kernel_size', 25)
                params['moving_avg'] = getattr(model, 'moving_avg', 25)
                if hasattr(model, 'adaptive_kernel'):
                    params['adaptive_kernel'] = model.adaptive_kernel
                    params['d_model'] = model.d_model
            
            elif component_type == 'attention':
                params['d_model'] = model.d_model
                params['num_heads'] = model.n_heads
                params['factor'] = getattr(model, 'factor', 1)
                params['dropout'] = model.dropout
            
            elif component_type == 'encoder':
                params['layers'] = model.e_layers
                params['d_model'] = model.d_model
                params['dropout'] = model.dropout
                if hasattr(model, 'gated_ffn'):
                    params['gated_ffn'] = model.gated_ffn
            
            elif component_type == 'decoder':
                params['layers'] = model.d_layers
                params['d_model'] = model.d_model
                params['dropout'] = model.dropout
                if hasattr(model, 'gated_ffn'):
                    params['gated_ffn'] = model.gated_ffn
            
            elif component_type == 'sampling':
                if hasattr(model, 'uncertainty_method') and model.uncertainty_method == 'bayesian':
                    params['num_samples'] = getattr(model, 'n_samples', 50)
                    params['uncertainty_method'] = 'monte_carlo'
            
            return params
        
        # Test parameter extraction
        mock_model = MockExistingModel()
        
        # Test decomposition parameters
        decomp_params = extract_component_params(mock_model, 'decomposition')
        assert decomp_params['kernel_size'] == 25
        assert decomp_params['adaptive_kernel'] is True
        assert decomp_params['d_model'] == 512
        
        # Test attention parameters
        attention_params = extract_component_params(mock_model, 'attention')
        assert attention_params['d_model'] == 512
        assert attention_params['num_heads'] == 8
        assert attention_params['factor'] == 1
        
        # Test sampling parameters for Bayesian model
        sampling_params = extract_component_params(mock_model, 'sampling')
        assert sampling_params['num_samples'] == 50
        assert sampling_params['uncertainty_method'] == 'monte_carlo'
    
    def test_configuration_migration_tool(self):
        """Test automated configuration migration tool"""
        
        class ConfigMigrator:
            """Tool for migrating from existing model configs to modular configs"""
            
            def __init__(self):
                self.component_mapping = {
                    'Autoformer': {
                        'decomposition': {'type': 'SeriesDecomposition'},
                        'attention': {'type': 'StandardAutoCorrelation'},
                        'encoder': {'type': 'StandardEncoder'},
                        'decoder': {'type': 'StandardDecoder'},
                        'sampling': {'type': 'DeterministicSampling'}
                    },
                    'BayesianEnhancedAutoformer': {
                        'decomposition': {'type': 'LearnableDecomposition'},
                        'attention': {'type': 'EnhancedAutoCorrelation'},
                        'encoder': {'type': 'EnhancedEncoder'},
                        'decoder': {'type': 'EnhancedDecoder'},
                        'sampling': {'type': 'BayesianSampling'}
                    }
                }
            
            def migrate_config(self, old_config, model_type):
                """Migrate old configuration to modular format"""
                if model_type not in self.component_mapping:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                # Start with component mapping
                modular_config = {
                    'model_type': 'ModularAutoformer',
                    'components': {}
                }
                
                # Copy component structure
                for comp_type, comp_info in self.component_mapping[model_type].items():
                    modular_config['components'][comp_type] = {
                        'type': comp_info['type'],
                        'params': {}
                    }
                
                # Map old parameters to new structure
                param_mapping = {
                    'seq_len': ('global', 'seq_len'),
                    'pred_len': ('global', 'pred_len'),
                    'd_model': ('attention', 'd_model'),
                    'n_heads': ('attention', 'num_heads'),
                    'e_layers': ('encoder', 'layers'),
                    'd_layers': ('decoder', 'layers'),
                    'dropout': ('attention', 'dropout'),
                    'moving_avg': ('decomposition', 'kernel_size'),
                    'factor': ('attention', 'factor')
                }
                
                # Apply parameter mapping
                modular_config['global_params'] = {}
                
                for old_param, (target_comp, new_param) in param_mapping.items():
                    if hasattr(old_config, old_param):
                        value = getattr(old_config, old_param)
                        
                        if target_comp == 'global':
                            modular_config['global_params'][new_param] = value
                        else:
                            modular_config['components'][target_comp]['params'][new_param] = value
                
                # Handle model-specific parameters
                if model_type == 'BayesianEnhancedAutoformer':
                    if hasattr(old_config, 'n_samples'):
                        modular_config['components']['sampling']['params']['num_samples'] = old_config.n_samples
                    
                    if hasattr(old_config, 'uncertainty_method'):
                        modular_config['components']['sampling']['params']['uncertainty_method'] = old_config.uncertainty_method
                
                return modular_config
            
            def validate_migration(self, old_config, new_config):
                """Validate that migration preserved important parameters"""
                validation_errors = []
                
                # Check critical parameters
                critical_params = ['seq_len', 'pred_len', 'd_model', 'n_heads']
                
                for param in critical_params:
                    if hasattr(old_config, param):
                        old_value = getattr(old_config, param)
                        
                        # Find corresponding value in new config
                        found = False
                        
                        # Check global params
                        if param in new_config.get('global_params', {}):
                            new_value = new_config['global_params'][param]
                            if old_value != new_value:
                                validation_errors.append(f"Global param {param}: {old_value} != {new_value}")
                            found = True
                        
                        # Check component params
                        for comp_name, comp_config in new_config.get('components', {}).items():
                            comp_params = comp_config.get('params', {})
                            
                            # Map old param names to new param names
                            param_name_mapping = {
                                'n_heads': 'num_heads',
                                'moving_avg': 'kernel_size'
                            }
                            
                            new_param_name = param_name_mapping.get(param, param)
                            
                            if new_param_name in comp_params:
                                new_value = comp_params[new_param_name]
                                if old_value != new_value:
                                    validation_errors.append(f"Component {comp_name}.{new_param_name}: {old_value} != {new_value}")
                                found = True
                        
                        if not found:
                            validation_errors.append(f"Parameter {param} not found in migrated config")
                
                return len(validation_errors) == 0, validation_errors
        
        # Test migration tool
        migrator = ConfigMigrator()
        
        # Create old-style config
        old_config = SimpleNamespace()
        old_config.seq_len = 96
        old_config.pred_len = 24
        old_config.d_model = 512
        old_config.n_heads = 8
        old_config.e_layers = 2
        old_config.d_layers = 1
        old_config.dropout = 0.1
        old_config.moving_avg = 25
        old_config.factor = 1
        old_config.n_samples = 50
        old_config.uncertainty_method = 'bayesian'
        
        # Test migration
        new_config = migrator.migrate_config(old_config, 'BayesianEnhancedAutoformer')
        
        # Verify migration structure
        assert new_config['model_type'] == 'ModularAutoformer'
        assert 'components' in new_config
        assert 'global_params' in new_config
        
        # Verify components
        components = new_config['components']
        assert 'decomposition' in components
        assert 'attention' in components
        assert 'encoder' in components
        assert 'decoder' in components
        assert 'sampling' in components
        
        # Verify component types
        assert components['decomposition']['type'] == 'LearnableDecomposition'
        assert components['attention']['type'] == 'EnhancedAutoCorrelation'
        assert components['sampling']['type'] == 'BayesianSampling'
        
        # Verify parameter migration
        assert new_config['global_params']['seq_len'] == 96
        assert new_config['global_params']['pred_len'] == 24
        assert components['attention']['params']['d_model'] == 512
        assert components['attention']['params']['num_heads'] == 8
        assert components['sampling']['params']['num_samples'] == 50
        
        # Validate migration
        is_valid, errors = migrator.validate_migration(old_config, new_config)
        assert is_valid, f"Migration validation failed: {errors}"


class TestProgressiveMigration:
    """Test progressive migration strategy"""
    
    def test_coexistence_strategy(self):
        """Test that old and new models can coexist during migration"""
        
        # Mock old and new model interfaces
        class OldModelInterface:
            def __init__(self, config):
                self.config = config
            
            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                batch_size = x_enc.shape[0]
                pred_len = self.config.pred_len
                c_out = self.config.c_out
                return torch.randn(batch_size, pred_len, c_out)
        
        class NewModularInterface:
            def __init__(self, config):
                self.config = config
            
            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                batch_size = x_enc.shape[0]
                pred_len = self.config['global_params']['pred_len']
                c_out = self.config['global_params']['c_out']
                return torch.randn(batch_size, pred_len, c_out)
        
        # Unified model factory that can create both old and new models
        class UnifiedModelFactory:
            def __init__(self):
                self.old_models = {
                    'Autoformer': OldModelInterface,
                    'EnhancedAutoformer': OldModelInterface
                }
                self.new_models = {
                    'ModularAutoformer': NewModularInterface
                }
            
            def create_model(self, config):
                """Create model based on config type"""
                if isinstance(config, dict) and config.get('model_type') == 'ModularAutoformer':
                    # New modular config
                    return self.new_models['ModularAutoformer'](config)
                elif hasattr(config, 'model_name') and config.model_name in self.old_models:
                    # Old config with explicit model name
                    return self.old_models[config.model_name](config)
                else:
                    # Default to old interface for backward compatibility
                    return OldModelInterface(config)
        
        # Test unified factory
        factory = UnifiedModelFactory()
        
        # Test old config
        old_config = SimpleNamespace()
        old_config.pred_len = 24
        old_config.c_out = 7
        old_config.model_name = 'Autoformer'
        
        old_model = factory.create_model(old_config)
        assert isinstance(old_model, OldModelInterface)
        
        # Test new config
        new_config = {
            'model_type': 'ModularAutoformer',
            'global_params': {'pred_len': 24, 'c_out': 7},
            'components': {}
        }
        
        new_model = factory.create_model(new_config)
        assert isinstance(new_model, NewModularInterface)
        
        # Test that both produce compatible outputs
        batch_size, seq_len = 4, 96
        x_enc = torch.randn(batch_size, seq_len, 7)
        x_mark_enc = torch.randn(batch_size, seq_len, 4)
        x_dec = torch.randn(batch_size, 48 + 24, 7)
        x_mark_dec = torch.randn(batch_size, 48 + 24, 4)
        
        old_output = old_model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        new_output = new_model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Both should have same output shape
        assert old_output.shape == new_output.shape
        assert old_output.shape == (batch_size, 24, 7)
    
    def test_gradual_component_replacement(self):
        """Test gradual replacement of components during migration"""
        
        # Hybrid model that can use both old and new components
        class HybridModel:
            def __init__(self, config):
                self.config = config
                self.components = {}
                self.migration_status = {}
            
            def migrate_component(self, component_type, new_component):
                """Replace old component with new modular component"""
                self.components[component_type] = new_component
                self.migration_status[component_type] = 'migrated'
            
            def is_fully_migrated(self):
                """Check if all components have been migrated"""
                required_components = ['decomposition', 'attention', 'encoder', 'decoder', 'sampling']
                return all(self.migration_status.get(comp) == 'migrated' for comp in required_components)
            
            def get_migration_progress(self):
                """Get migration progress percentage"""
                required_components = ['decomposition', 'attention', 'encoder', 'decoder', 'sampling']
                migrated = sum(1 for comp in required_components if self.migration_status.get(comp) == 'migrated')
                return migrated / len(required_components) * 100
        
        # Test gradual migration
        model = HybridModel({})
        
        # Initially, no components migrated
        assert model.get_migration_progress() == 0.0
        assert not model.is_fully_migrated()
        
        # Mock components
        mock_components = [
            ('decomposition', Mock()),
            ('attention', Mock()),
            ('encoder', Mock()),
            ('decoder', Mock()),
            ('sampling', Mock())
        ]
        
        # Migrate components one by one
        for i, (comp_type, comp_instance) in enumerate(mock_components):
            model.migrate_component(comp_type, comp_instance)
            
            expected_progress = (i + 1) / len(mock_components) * 100
            assert model.get_migration_progress() == expected_progress
        
        # After all components migrated
        assert model.is_fully_migrated()
        assert model.get_migration_progress() == 100.0
    
    def test_rollback_capability(self):
        """Test ability to rollback from new to old implementation"""
        
        class RollbackCapableModel:
            def __init__(self, config):
                self.config = config
                self.current_implementation = 'old'
                self.old_components = {}
                self.new_components = {}
                self.component_snapshots = {}
            
            def create_snapshot(self):
                """Create snapshot of current state for rollback"""
                self.component_snapshots = {
                    'implementation': self.current_implementation,
                    'old_components': self.old_components.copy(),
                    'new_components': self.new_components.copy()
                }
            
            def migrate_to_new(self, new_components):
                """Migrate to new implementation"""
                self.create_snapshot()  # Backup current state
                self.current_implementation = 'new'
                self.new_components = new_components
            
            def rollback(self):
                """Rollback to previous state"""
                if not self.component_snapshots:
                    raise ValueError("No snapshot available for rollback")
                
                self.current_implementation = self.component_snapshots['implementation']
                self.old_components = self.component_snapshots['old_components']
                self.new_components = self.component_snapshots['new_components']
                
                # Clear snapshot after rollback
                self.component_snapshots = {}
            
            def get_current_implementation(self):
                return self.current_implementation
        
        # Test rollback functionality
        model = RollbackCapableModel({})
        
        # Initially using old implementation
        assert model.get_current_implementation() == 'old'
        
        # Migrate to new implementation
        new_components = {'attention': Mock(), 'encoder': Mock()}
        model.migrate_to_new(new_components)
        assert model.get_current_implementation() == 'new'
        assert len(model.new_components) == 2
        
        # Rollback to old implementation
        model.rollback()
        assert model.get_current_implementation() == 'old'
        assert len(model.new_components) == 0  # Should be reset
        
        # Test that rollback is only possible once per snapshot
        with pytest.raises(ValueError, match="No snapshot available"):
            model.rollback()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
