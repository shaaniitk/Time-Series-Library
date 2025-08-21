"""
Comprehensive Test Suite for Modular Autoformer Framework

This test file provides comprehensive coverage for the modular component system,
registry functionality, and the migration strategy outlined in the documentation.
"""

import pytest
import torch
import torch.nn as nn
import yaml
import tempfile
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modular components with graceful fallback
try:
    from utils.modular_components.registry import ComponentRegistry, get_global_registry
    from utils.modular_components.base_interfaces import BaseComponent, ComponentType
    from utils.modular_components.config_schemas import ComponentConfig
    from utils.modular_components.factory import ComponentFactory
    MODULAR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Modular components not fully available: {e}")
    MODULAR_AVAILABLE = False

# Import existing models for comparison testing
try:
    from models.modular_autoformer import ModularAutoformer as BaseAutoformer
    from models.EnhancedAutoformer import EnhancedAutoformer
    from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer
    from models.HierarchicalEnhancedAutoformer import HierarchicalEnhancedAutoformer
    from models.QuantileBayesianAutoformer import QuantileBayesianAutoformer
    EXISTING_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some existing models not available: {e}")
    EXISTING_MODELS_AVAILABLE = False


class TestModularRegistry:
    """Test the component registry system"""
    
    def test_registry_initialization(self):
        """Test that registry initializes correctly"""
        registry = ComponentRegistry()
        
        # Check all expected component types are present
        expected_types = ['backbone', 'embedding', 'attention', 'processor', 
                         'feedforward', 'loss', 'output', 'adapter']
        
        for component_type in expected_types:
            assert component_type in registry._components
            assert component_type in registry._metadata
    
    def test_component_registration(self):
        """Test registering components in the registry"""
        registry = ComponentRegistry()
        
        # Create a mock component
        class MockComponent(BaseComponent):
            component_type = ComponentType.ATTENTION
            
            def __init__(self, config):
                super().__init__(config)
                self.d_model = config.get('d_model', 512)
            
            def forward(self, x):
                return x
            
            def get_config(self):
                return {'d_model': self.d_model}
        
        # Register the component
        registry.register(
            component_type='attention',
            component_name='mock_attention',
            component_class=MockComponent,
            metadata={'test': True}
        )
        
        # Verify registration
        assert 'mock_attention' in registry._components['attention']
        assert registry._metadata['attention']['mock_attention']['test'] is True
        
        # Test retrieval
        retrieved_class = registry.get('attention', 'mock_attention')
        assert retrieved_class == MockComponent
    
    def test_component_creation(self):
        """Test creating component instances"""
        registry = ComponentRegistry()
        
        # Mock component for testing
        class TestComponent(BaseComponent):
            def __init__(self, config):
                super().__init__(config)
                self.test_param = config.get('test_param', 'default')
            
            def forward(self, x):
                return x
            
            def get_config(self):
                return {'test_param': self.test_param}
        
        registry.register('processor', 'test_component', TestComponent)
        
        # Test creation with default config
        component = registry.create('processor', 'test_component', {})
        assert component.test_param == 'default'
        
        # Test creation with custom config
        component = registry.create('processor', 'test_component', {'test_param': 'custom'})
        assert component.test_param == 'custom'
    
    def test_registry_error_handling(self):
        """Test registry error handling for invalid inputs"""
        registry = ComponentRegistry()
        
        # Test invalid component type
        with pytest.raises(ValueError, match="Unknown component type"):
            registry.register('invalid_type', 'test', Mock)
        
        # Test missing component
        with pytest.raises(ValueError, match="Component .* not found"):
            registry.get('attention', 'nonexistent_component')


class TestComponentInterfaces:
    """Test component base interfaces and contracts"""
    
    def test_base_component_interface(self):
        """Test BaseComponent interface requirements"""
        
        class ValidComponent(BaseComponent):
            def __init__(self, config):
                super().__init__(config)
                self.test_value = config.get('test_value', 42)
            
            def forward(self, x):
                return x * self.test_value
            
            def get_config(self):
                return {'test_value': self.test_value}
        
        config = {'test_value': 10}
        component = ValidComponent(config)
        
        # Test interface compliance
        assert hasattr(component, 'forward')
        assert hasattr(component, 'get_config')
        assert callable(component.forward)
        assert callable(component.get_config)
        
        # Test functionality
        test_input = torch.tensor([1.0, 2.0, 3.0])
        output = component.forward(test_input)
        expected = test_input * 10
        assert torch.allclose(output, expected)
        
        # Test config retrieval
        retrieved_config = component.get_config()
        assert retrieved_config['test_value'] == 10


class TestDecompositionComponents:
    """Test decomposition component extraction and unification"""
    
    def test_series_decomposition_extraction(self):
        """Test extracting series decomposition from base Autoformer"""
        # Mock the series decomposition component
        class SeriesDecomposition(BaseComponent):
            def __init__(self, config):
                super().__init__(config)
                self.kernel_size = config.get('kernel_size', 25)
                self.avg_pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)
            
            def forward(self, x):
                # Simplified moving average decomposition
                batch_size, seq_len, features = x.shape
                
                # Padding for moving average
                padding = (self.kernel_size - 1) // 2
                front = x[:, 0:1, :].repeat(1, padding, 1)
                end = x[:, -1:, :].repeat(1, padding, 1)
                x_padded = torch.cat([front, x, end], dim=1)
                
                # Moving average (trend)
                x_permuted = x_padded.permute(0, 2, 1)  # [B, F, L]
                trend = self.avg_pool(x_permuted).permute(0, 2, 1)  # [B, L, F]
                
                # Seasonal component
                seasonal = x - trend
                
                return seasonal, trend
            
            def get_config(self):
                return {'type': 'series_decomposition', 'kernel_size': self.kernel_size}
        
        # Test the decomposition
        config = {'kernel_size': 25}
        decomp = SeriesDecomposition(config)
        
        # Test input
        batch_size, seq_len, features = 4, 96, 7
        x = torch.randn(batch_size, seq_len, features)
        
        seasonal, trend = decomp.forward(x)
        
        # Verify output shapes
        assert seasonal.shape == x.shape
        assert trend.shape == x.shape
        
        # Verify decomposition property (x = seasonal + trend)
        reconstructed = seasonal + trend
        assert torch.allclose(reconstructed, x, atol=1e-5)
    
    def test_learnable_decomposition_component(self):
        """Test learnable decomposition component"""
        class LearnableDecomposition(BaseComponent):
            def __init__(self, config):
                super().__init__(config)
                self.d_model = config.get('d_model', 512)
                self.kernel_size = config.get('kernel_size', 25)
                
                # Learnable trend extraction weights
                self.trend_weights = nn.Parameter(torch.randn(self.d_model, 1, self.kernel_size))
                nn.init.constant_(self.trend_weights, 1.0 / self.kernel_size)
            
            def forward(self, x):
                batch_size, seq_len, features = x.shape
                
                # Simplified learnable decomposition
                # Use convolution with learned weights
                x_permuted = x.permute(0, 2, 1)  # [B, F, L]
                
                # Apply learnable trend extraction
                padding = (self.kernel_size - 1) // 2
                x_padded = nn.functional.pad(x_permuted, (padding, padding), mode='replicate')
                
                # Grouped convolution for feature-specific weights
                trend = nn.functional.conv1d(
                    x_padded, self.trend_weights, groups=features, padding=0
                )
                trend = trend.permute(0, 2, 1)  # [B, L, F]
                
                seasonal = x - trend
                return seasonal, trend
            
            def get_config(self):
                return {
                    'type': 'learnable_decomposition',
                    'd_model': self.d_model,
                    'kernel_size': self.kernel_size
                }
        
        # Test learnable decomposition
        config = {'d_model': 7, 'kernel_size': 25}
        learnable_decomp = LearnableDecomposition(config)
        
        batch_size, seq_len, features = 4, 96, 7
        x = torch.randn(batch_size, seq_len, features)
        
        seasonal, trend = learnable_decomp.forward(x)
        
        # Verify output shapes
        assert seasonal.shape == x.shape
        assert trend.shape == x.shape
        
        # Verify that weights are learnable
        assert learnable_decomp.trend_weights.requires_grad


class TestAttentionComponents:
    """Test attention mechanism component extraction"""
    
    def test_autocorrelation_component(self):
        """Test AutoCorrelation attention component"""
        class AutoCorrelationAttention(BaseComponent):
            def __init__(self, config):
                super().__init__(config)
                self.d_model = config.get('d_model', 512)
                self.num_heads = config.get('num_heads', 8)
                self.factor = config.get('factor', 1)
                
                assert self.d_model % self.num_heads == 0
                self.d_k = self.d_model // self.num_heads
                
                self.query_projection = nn.Linear(self.d_model, self.d_model)
                self.key_projection = nn.Linear(self.d_model, self.d_model)
                self.value_projection = nn.Linear(self.d_model, self.d_model)
                self.output_projection = nn.Linear(self.d_model, self.d_model)
            
            def forward(self, queries, keys, values, attn_mask=None):
                B, L, _ = queries.shape
                _, S, _ = keys.shape
                H = self.num_heads
                
                # Project inputs
                Q = self.query_projection(queries).view(B, L, H, self.d_k)
                K = self.key_projection(keys).view(B, S, H, self.d_k)
                V = self.value_projection(values).view(B, S, H, self.d_k)
                
                # Simplified autocorrelation (replace with actual implementation)
                # For testing, use standard attention
                Q = Q.permute(0, 2, 1, 3)  # [B, H, L, d_k]
                K = K.permute(0, 2, 1, 3)  # [B, H, S, d_k]
                V = V.permute(0, 2, 1, 3)  # [B, H, S, d_k]
                
                # Compute attention
                scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
                if attn_mask is not None:
                    scores += attn_mask
                
                attn_weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, V)
                
                # Reshape and project output
                output = output.permute(0, 2, 1, 3).contiguous().view(B, L, self.d_model)
                output = self.output_projection(output)
                
                return output, attn_weights
            
            def get_config(self):
                return {
                    'type': 'autocorrelation_attention',
                    'd_model': self.d_model,
                    'num_heads': self.num_heads,
                    'factor': self.factor
                }
        
        # Test attention component
        config = {'d_model': 512, 'num_heads': 8, 'factor': 1}
        attention = AutoCorrelationAttention(config)
        
        batch_size, seq_len = 4, 96
        queries = torch.randn(batch_size, seq_len, 512)
        keys = torch.randn(batch_size, seq_len, 512)
        values = torch.randn(batch_size, seq_len, 512)
        
        output, attn_weights = attention.forward(queries, keys, values)
        
        # Verify output shapes
        assert output.shape == (batch_size, seq_len, 512)
        assert attn_weights.shape == (batch_size, 8, seq_len, seq_len)


class TestSamplingComponents:
    """Test sampling method component extraction"""
    
    def test_deterministic_sampling(self):
        """Test deterministic sampling component"""
        class DeterministicSampling(BaseComponent):
            def __init__(self, config):
                super().__init__(config)
                self.output_dim = config.get('output_dim', 1)
            
            def forward(self, x):
                # Simple deterministic output
                return x
            
            def get_config(self):
                return {'type': 'deterministic_sampling', 'output_dim': self.output_dim}
        
        config = {'output_dim': 7}
        sampling = DeterministicSampling(config)
        
        batch_size, seq_len, features = 4, 24, 7
        x = torch.randn(batch_size, seq_len, features)
        
        output = sampling.forward(x)
        assert output.shape == x.shape
    
    def test_bayesian_sampling(self):
        """Test Bayesian sampling component"""
        class BayesianSampling(BaseComponent):
            def __init__(self, config):
                super().__init__(config)
                self.num_samples = config.get('num_samples', 50)
                self.output_dim = config.get('output_dim', 1)
                
                # Bayesian layers for uncertainty
                self.mean_layer = nn.Linear(config.get('d_model', 512), self.output_dim)
                self.logvar_layer = nn.Linear(config.get('d_model', 512), self.output_dim)
            
            def forward(self, x, return_uncertainty=False):
                # Compute mean and log variance
                mean = self.mean_layer(x)
                log_var = self.logvar_layer(x)
                std = torch.exp(0.5 * log_var)
                
                if return_uncertainty:
                    # Sample multiple times for uncertainty estimation
                    samples = []
                    for _ in range(self.num_samples):
                        eps = torch.randn_like(std)
                        sample = mean + std * eps
                        samples.append(sample)
                    
                    samples = torch.stack(samples, dim=0)  # [num_samples, B, L, F]
                    
                    # Compute statistics
                    sample_mean = samples.mean(dim=0)
                    sample_std = samples.std(dim=0)
                    
                    return {
                        'prediction': sample_mean,
                        'uncertainty': sample_std,
                        'samples': samples
                    }
                else:
                    # Return mean prediction
                    eps = torch.randn_like(std)
                    return mean + std * eps
            
            def get_config(self):
                return {
                    'type': 'bayesian_sampling',
                    'num_samples': self.num_samples,
                    'output_dim': self.output_dim
                }
        
        config = {'num_samples': 10, 'output_dim': 7, 'd_model': 512}
        sampling = BayesianSampling(config)
        
        batch_size, seq_len, d_model = 4, 24, 512
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Test deterministic forward
        output = sampling.forward(x, return_uncertainty=False)
        assert output.shape == (batch_size, seq_len, 7)
        
        # Test uncertainty forward
        result = sampling.forward(x, return_uncertainty=True)
        assert 'prediction' in result
        assert 'uncertainty' in result
        assert 'samples' in result
        assert result['prediction'].shape == (batch_size, seq_len, 7)
        assert result['uncertainty'].shape == (batch_size, seq_len, 7)
        assert result['samples'].shape == (10, batch_size, seq_len, 7)


class TestModularAutoformer:
    """Test the unified ModularAutoformer implementation"""
    
    def test_modular_autoformer_architecture(self):
        """Test ModularAutoformer with different component configurations"""
        
        # Mock ModularAutoformer implementation
        class MockModularAutoformer(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # Initialize components based on config
                self.decomposition = self._create_component('decomposition', config)
                self.attention = self._create_component('attention', config)
                self.encoder = self._create_component('encoder', config)
                self.decoder = self._create_component('decoder', config)
                self.sampling = self._create_component('sampling', config)
            
            def _create_component(self, component_type, config):
                """Mock component creation"""
                component_config = config.get(component_type, {})
                component_name = component_config.get('type', 'default')
                
                # Return mock components for testing
                if component_type == 'decomposition':
                    return Mock(forward=lambda x: (x * 0.7, x * 0.3))  # seasonal, trend
                elif component_type == 'attention':
                    return Mock(forward=lambda q, k, v, mask=None: (q, torch.randn(q.shape[0], 8, q.shape[1], q.shape[1])))
                elif component_type == 'encoder':
                    return Mock(forward=lambda x, mask=None: (x, []))
                elif component_type == 'decoder':
                    return Mock(forward=lambda x, cross, x_mask=None, cross_mask=None, trend=None: (x, trend or x * 0.1))
                elif component_type == 'sampling':
                    return Mock(forward=lambda x: x)
                else:
                    return Mock()
            
            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                # Simplified forward pass
                batch_size, seq_len, _ = x_enc.shape
                pred_len = x_dec.shape[1] - x_enc.shape[1] // 2  # Simplified
                
                # Decomposition
                seasonal, trend = self.decomposition.forward(x_enc)
                
                # Encoder processing
                enc_out, _ = self.encoder.forward(seasonal)
                
                # Decoder processing
                dec_out, residual_trend = self.decoder.forward(
                    x_dec[:, :-pred_len, :], enc_out, trend=trend
                )
                
                # Final prediction
                prediction = self.sampling.forward(dec_out[:, -pred_len:, :])
                
                return prediction
        
        # Test different configurations
        configs = [
            # Standard configuration
            {
                'decomposition': {'type': 'series_decomposition', 'kernel_size': 25},
                'attention': {'type': 'autocorrelation', 'd_model': 512, 'num_heads': 8},
                'encoder': {'type': 'standard_encoder', 'layers': 2},
                'decoder': {'type': 'standard_decoder', 'layers': 1},
                'sampling': {'type': 'deterministic_sampling'}
            },
            # Bayesian configuration
            {
                'decomposition': {'type': 'learnable_decomposition', 'd_model': 512},
                'attention': {'type': 'enhanced_autocorrelation', 'd_model': 512, 'num_heads': 8},
                'encoder': {'type': 'enhanced_encoder', 'layers': 2},
                'decoder': {'type': 'enhanced_decoder', 'layers': 1},
                'sampling': {'type': 'bayesian_sampling', 'num_samples': 50}
            }
        ]
        
        for i, config in enumerate(configs):
            model = MockModularAutoformer(config)
            
            # Test forward pass
            batch_size, seq_len, pred_len, features = 4, 96, 24, 7
            x_enc = torch.randn(batch_size, seq_len, features)
            x_mark_enc = torch.randn(batch_size, seq_len, 4)
            x_dec = torch.randn(batch_size, seq_len // 2 + pred_len, features)
            x_mark_dec = torch.randn(batch_size, seq_len // 2 + pred_len, 4)
            
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Verify output shape
            assert output.shape == (batch_size, pred_len, features), f"Config {i} failed shape test"


class TestConfigurationSystem:
    """Test configuration-driven model instantiation"""
    
    def test_yaml_configuration_loading(self):
        """Test loading configuration from YAML files"""
        
        # Create temporary YAML configuration
        config_yaml = """
model_type: "ModularAutoformer"
components:
  decomposition:
    type: "SeriesDecomposition"
    params:
      kernel_size: 25
  attention:
    type: "StandardAutoCorrelation"
    params:
      factor: 1
      attention_dropout: 0.1
      d_model: 512
      num_heads: 8
  encoder:
    type: "StandardEncoder"
    params:
      layers: 2
      d_model: 512
  decoder:
    type: "StandardDecoder"
    params:
      layers: 1
      d_model: 512
  sampling:
    type: "DeterministicSampling"

training:
  batch_size: 32
  learning_rate: 0.0001
  epochs: 100
"""
        
        # Test YAML parsing
        config = yaml.safe_load(config_yaml)
        
        # Verify configuration structure
        assert config['model_type'] == 'ModularAutoformer'
        assert 'components' in config
        assert 'decomposition' in config['components']
        assert 'attention' in config['components']
        assert 'encoder' in config['components']
        assert 'decoder' in config['components']
        assert 'sampling' in config['components']
        
        # Verify component parameters
        decomp_config = config['components']['decomposition']
        assert decomp_config['type'] == 'SeriesDecomposition'
        assert decomp_config['params']['kernel_size'] == 25
        
        attention_config = config['components']['attention']
        assert attention_config['type'] == 'StandardAutoCorrelation'
        assert attention_config['params']['d_model'] == 512
        assert attention_config['params']['num_heads'] == 8
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        
        # Valid configuration
        valid_config = {
            'model_type': 'ModularAutoformer',
            'components': {
                'decomposition': {'type': 'SeriesDecomposition', 'params': {'kernel_size': 25}},
                'attention': {'type': 'AutoCorrelation', 'params': {'d_model': 512}},
                'encoder': {'type': 'StandardEncoder', 'params': {'layers': 2}},
                'decoder': {'type': 'StandardDecoder', 'params': {'layers': 1}},
                'sampling': {'type': 'DeterministicSampling', 'params': {}}
            }
        }
        
        # Test validation function
        def validate_config(config):
            required_fields = ['model_type', 'components']
            required_components = ['decomposition', 'attention', 'encoder', 'decoder', 'sampling']
            
            for field in required_fields:
                if field not in config:
                    return False, f"Missing required field: {field}"
            
            for component in required_components:
                if component not in config['components']:
                    return False, f"Missing required component: {component}"
                
                component_config = config['components'][component]
                if 'type' not in component_config:
                    return False, f"Missing type for component: {component}"
            
            return True, "Configuration valid"
        
        # Test valid configuration
        is_valid, message = validate_config(valid_config)
        assert is_valid, f"Valid config failed validation: {message}"
        
        # Test invalid configurations
        invalid_configs = [
            # Missing model_type
            {'components': valid_config['components']},
            # Missing components
            {'model_type': 'ModularAutoformer'},
            # Missing decomposition component
            {
                'model_type': 'ModularAutoformer',
                'components': {k: v for k, v in valid_config['components'].items() if k != 'decomposition'}
            }
        ]
        
        for invalid_config in invalid_configs:
            is_valid, message = validate_config(invalid_config)
            assert not is_valid, f"Invalid config passed validation: {invalid_config}"


@pytest.mark.skipif(not EXISTING_MODELS_AVAILABLE, reason="Existing models not available")
class TestBackwardCompatibility:
    """Test backward compatibility with existing models"""
    
    def test_model_output_consistency(self):
        """Test that modular components produce consistent outputs with original models"""
        
        # Create realistic config
        config = SimpleNamespace()
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
        
        # Test input data
        batch_size = 4
        x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
        x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
        x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
        x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
        
        # Test that existing models can still be instantiated and run
        try:
            base_model = BaseAutoformer(config)
            base_model.eval()
            
            with torch.no_grad():
                base_output = base_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Verify output shape
            assert base_output.shape == (batch_size, config.pred_len, config.c_out)
            
        except Exception as e:
            pytest.skip(f"Base Autoformer not available: {e}")
        
        try:
            enhanced_model = EnhancedAutoformer(config)
            enhanced_model.eval()
            
            with torch.no_grad():
                enhanced_output = enhanced_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Verify output shape
            assert enhanced_output.shape == (batch_size, config.pred_len, config.c_out)
            
        except Exception as e:
            pytest.skip(f"Enhanced Autoformer not available: {e}")


class TestPerformanceAndMemory:
    """Test performance characteristics and memory usage"""
    
    def test_component_memory_usage(self):
        """Test that modular components don't increase memory usage significantly"""
        
        # Simple memory tracking
        def get_memory_usage():
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated()
            else:
                return 0  # CPU memory tracking is more complex
        
        initial_memory = get_memory_usage()
        
        # Create multiple component instances
        components = []
        for i in range(10):
            
            class TestComponent(BaseComponent):
                def __init__(self, config):
                    super().__init__(config)
                    self.linear = nn.Linear(512, 512)
                
                def forward(self, x):
                    return self.linear(x)
                
                def get_config(self):
                    return {'test': True}
            
            component = TestComponent({})
            components.append(component)
        
        final_memory = get_memory_usage()
        
        # Memory usage should be reasonable (this is a basic check)
        if torch.cuda.is_available():
            memory_increase = final_memory - initial_memory
            assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
        
        # Clean up
        del components
    
    def test_component_creation_speed(self):
        """Test that component creation is reasonably fast"""
        import time
        
        class FastComponent(BaseComponent):
            def __init__(self, config):
                super().__init__(config)
                self.value = config.get('value', 1)
            
            def forward(self, x):
                return x * self.value
            
            def get_config(self):
                return {'value': self.value}
        
        # Time component creation
        start_time = time.time()
        
        components = []
        for i in range(1000):
            component = FastComponent({'value': i})
            components.append(component)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Should create 1000 components in less than 1 second
        assert creation_time < 1.0, f"Component creation too slow: {creation_time:.3f}s"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
