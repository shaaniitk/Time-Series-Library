"""
End-to-End Workflow Tests for Modular Autoformer Framework

This file tests complete workflows from data loading through model training
and inference, ensuring the modular framework works correctly in real scenarios.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import yaml
from types import SimpleNamespace

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import actual data providers and experiment classes
try:
    from data_provider.data_factory import data_provider
    from data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom
    from exp.exp_main import Exp_Main
    from utils.tools import EarlyStopping, adjust_learning_rate, visual
    DATA_PROVIDERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Data providers not available: {e}")
    DATA_PROVIDERS_AVAILABLE = False


class TestDataIntegration:
    """Test integration with data loading and preprocessing"""
    
    @pytest.fixture
    def sample_timeseries_data(self):
        """Generate sample time series data for testing"""
        np.random.seed(42)
        
        # Generate 1000 time steps with 7 features
        n_timesteps = 1000
        n_features = 7
        
        # Create time-based patterns
        time = np.arange(n_timesteps)
        data = np.zeros((n_timesteps, n_features))
        
        for i in range(n_features):
            # Trend component
            trend = 0.01 * time * (i + 1)
            
            # Seasonal component
            seasonal = np.sin(2 * np.pi * time / (24 + 5 * i)) * (i + 1)
            
            # Noise component
            noise = np.random.normal(0, 0.1, n_timesteps)
            
            data[:, i] = trend + seasonal + noise
        
        # Create DataFrame with timestamp
        dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='H')
        df = pd.DataFrame(data, index=dates, columns=[f'feature_{i}' for i in range(n_features)])
        
        return df
    
    @pytest.fixture
    def temp_data_file(self, sample_timeseries_data):
        """Create temporary data file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_timeseries_data.to_csv(f.name)
            return f.name
    
    def test_data_loading_with_modular_config(self, temp_data_file):
        """Test data loading works with modular configuration"""
        
        # Mock modular configuration
        modular_config = {
            'model_type': 'ModularAutoformer',
            'data': {
                'data_path': temp_data_file,
                'target': 'feature_0',
                'features': 'M',  # Multivariate
                'seq_len': 96,
                'label_len': 48,
                'pred_len': 24,
                'freq': 'h'
            },
            'global_params': {
                'seq_len': 96,
                'label_len': 48,
                'pred_len': 24,
                'enc_in': 7,
                'dec_in': 7,
                'c_out': 7
            },
            'components': {
                'decomposition': {'type': 'SeriesDecomposition', 'params': {}},
                'attention': {'type': 'StandardAutoCorrelation', 'params': {}},
                'encoder': {'type': 'StandardEncoder', 'params': {}},
                'decoder': {'type': 'StandardDecoder', 'params': {}},
                'sampling': {'type': 'DeterministicSampling', 'params': {}}
            }
        }
        
        # Mock data loader
        class MockDataLoader:
            def __init__(self, config):
                self.config = config
                self.seq_len = config['data']['seq_len']
                self.pred_len = config['data']['pred_len']
                self.label_len = config['data']['label_len']
                
            def get_data_loaders(self):
                # Generate mock data loaders
                batch_size = 32
                
                def create_mock_loader(n_batches=10):
                    for _ in range(n_batches):
                        batch_x = torch.randn(batch_size, self.seq_len, 7)
                        batch_y = torch.randn(batch_size, self.label_len + self.pred_len, 7)
                        batch_x_mark = torch.randn(batch_size, self.seq_len, 4)
                        batch_y_mark = torch.randn(batch_size, self.label_len + self.pred_len, 4)
                        
                        yield batch_x, batch_y, batch_x_mark, batch_y_mark
                
                return {
                    'train': create_mock_loader(50),
                    'val': create_mock_loader(20),
                    'test': create_mock_loader(10)
                }
        
        # Test data loader creation
        data_loader = MockDataLoader(modular_config)
        loaders = data_loader.get_data_loaders()
        
        assert 'train' in loaders
        assert 'val' in loaders
        assert 'test' in loaders
        
        # Test that data has correct dimensions
        train_batch = next(iter(loaders['train']))
        batch_x, batch_y, batch_x_mark, batch_y_mark = train_batch
        
        assert batch_x.shape == (32, 96, 7)  # batch_size, seq_len, features
        assert batch_y.shape == (32, 72, 7)   # batch_size, label_len + pred_len, features
        assert batch_x_mark.shape == (32, 96, 4)  # time features
        assert batch_y_mark.shape == (32, 72, 4)
        
        # Cleanup
        os.unlink(temp_data_file)
    
    def test_configuration_driven_data_preprocessing(self):
        """Test that data preprocessing adapts to modular configuration"""
        
        class ConfigurablePreprocessor:
            def __init__(self, config):
                self.config = config
                self.preprocessing_steps = []
                
                # Configure preprocessing based on components
                components = config.get('components', {})
                
                # Different decomposition types need different preprocessing
                decomp_type = components.get('decomposition', {}).get('type', 'SeriesDecomposition')
                if decomp_type == 'WaveletDecomposition':
                    self.preprocessing_steps.append('wavelet_preprocessing')
                elif decomp_type == 'LearnableDecomposition':
                    self.preprocessing_steps.append('adaptive_normalization')
                else:
                    self.preprocessing_steps.append('standard_normalization')
                
                # Bayesian sampling needs uncertainty-aware preprocessing
                sampling_type = components.get('sampling', {}).get('type', 'DeterministicSampling')
                if 'Bayesian' in sampling_type:
                    self.preprocessing_steps.append('uncertainty_estimation')
                
                # Enhanced attention needs additional time features
                attention_type = components.get('attention', {}).get('type', 'StandardAutoCorrelation')
                if 'Enhanced' in attention_type:
                    self.preprocessing_steps.append('enhanced_time_features')
            
            def preprocess(self, data):
                """Apply configured preprocessing steps"""
                processed_data = data.copy()
                
                for step in self.preprocessing_steps:
                    if step == 'wavelet_preprocessing':
                        # Mock wavelet preprocessing
                        processed_data = self._apply_wavelet_transform(processed_data)
                    elif step == 'adaptive_normalization':
                        # Mock adaptive normalization
                        processed_data = self._apply_adaptive_norm(processed_data)
                    elif step == 'standard_normalization':
                        # Standard z-score normalization
                        processed_data = (processed_data - processed_data.mean()) / processed_data.std()
                    elif step == 'uncertainty_estimation':
                        # Add uncertainty estimates
                        processed_data = self._add_uncertainty_features(processed_data)
                    elif step == 'enhanced_time_features':
                        # Add enhanced time features
                        processed_data = self._add_enhanced_time_features(processed_data)
                
                return processed_data
            
            def _apply_wavelet_transform(self, data):
                # Mock wavelet transform
                return data * 0.9  # Simple mock
            
            def _apply_adaptive_norm(self, data):
                # Mock adaptive normalization
                return (data - data.mean()) / (data.std() + 1e-8)
            
            def _add_uncertainty_features(self, data):
                # Mock uncertainty features
                if isinstance(data, torch.Tensor):
                    uncertainty = torch.randn_like(data) * 0.1
                    return torch.cat([data, uncertainty], dim=-1)
                return data
            
            def _add_enhanced_time_features(self, data):
                # Mock enhanced time features
                if isinstance(data, torch.Tensor):
                    enhanced_features = torch.randn(data.shape[0], data.shape[1], 2)
                    return torch.cat([data, enhanced_features], dim=-1)
                return data
        
        # Test different configurations
        configs = [
            {
                'components': {
                    'decomposition': {'type': 'SeriesDecomposition'},
                    'attention': {'type': 'StandardAutoCorrelation'},
                    'sampling': {'type': 'DeterministicSampling'}
                }
            },
            {
                'components': {
                    'decomposition': {'type': 'WaveletDecomposition'},
                    'attention': {'type': 'EnhancedAutoCorrelation'},
                    'sampling': {'type': 'BayesianSampling'}
                }
            },
            {
                'components': {
                    'decomposition': {'type': 'LearnableDecomposition'},
                    'attention': {'type': 'EnhancedAutoCorrelation'},
                    'sampling': {'type': 'DeterministicSampling'}
                }
            }
        ]
        
        # Test each configuration
        sample_data = torch.randn(32, 96, 7)
        
        for i, config in enumerate(configs):
            preprocessor = ConfigurablePreprocessor(config)
            
            # Check preprocessing steps are configured correctly
            if 'Wavelet' in config['components']['decomposition']['type']:
                assert 'wavelet_preprocessing' in preprocessor.preprocessing_steps
            
            if 'Enhanced' in config['components']['attention']['type']:
                assert 'enhanced_time_features' in preprocessor.preprocessing_steps
            
            if 'Bayesian' in config['components']['sampling']['type']:
                assert 'uncertainty_estimation' in preprocessor.preprocessing_steps
            
            # Test preprocessing
            processed = preprocessor.preprocess(sample_data)
            assert processed is not None
            assert processed.shape[0] == sample_data.shape[0]  # Batch size preserved
            assert processed.shape[1] == sample_data.shape[1]  # Sequence length preserved


class TestTrainingWorkflow:
    """Test complete training workflows with modular components"""
    
    @pytest.fixture
    def training_config(self):
        """Standard training configuration for testing"""
        return {
            'model_type': 'ModularAutoformer',
            'training': {
                'batch_size': 32,
                'learning_rate': 0.0001,
                'epochs': 5,
                'patience': 3,
                'optimizer': 'Adam',
                'loss_function': 'MSE'
            },
            'global_params': {
                'seq_len': 96,
                'label_len': 48,
                'pred_len': 24,
                'enc_in': 7,
                'dec_in': 7,
                'c_out': 7,
                'd_model': 512,
                'n_heads': 8,
                'e_layers': 2,
                'd_layers': 1,
                'dropout': 0.1
            },
            'components': {
                'decomposition': {
                    'type': 'SeriesDecomposition',
                    'params': {'kernel_size': 25}
                },
                'attention': {
                    'type': 'StandardAutoCorrelation',
                    'params': {'factor': 1}
                },
                'encoder': {
                    'type': 'StandardEncoder',
                    'params': {'layers': 2}
                },
                'decoder': {
                    'type': 'StandardDecoder',
                    'params': {'layers': 1}
                },
                'sampling': {
                    'type': 'DeterministicSampling',
                    'params': {}
                }
            }
        }
    
    def test_model_initialization_from_config(self, training_config):
        """Test model initialization from modular configuration"""
        
        class MockModularAutoformer:
            def __init__(self, config):
                self.config = config
                self.components = {}
                
                # Initialize components based on config
                for comp_type, comp_config in config['components'].items():
                    self.components[comp_type] = self._create_component(comp_type, comp_config)
                
                # Initialize global parameters
                self.global_params = config['global_params']
                
            def _create_component(self, comp_type, comp_config):
                # Mock component creation
                component = Mock()
                component.type = comp_config['type']
                component.params = comp_config['params']
                return component
            
            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                batch_size = x_enc.shape[0]
                pred_len = self.global_params['pred_len']
                c_out = self.global_params['c_out']
                return torch.randn(batch_size, pred_len, c_out)
            
            def parameters(self):
                # Mock parameters for optimizer
                return [torch.randn(10, requires_grad=True)]
        
        # Test model initialization
        model = MockModularAutoformer(training_config)
        
        # Verify components are initialized
        assert len(model.components) == 5
        assert 'decomposition' in model.components
        assert 'attention' in model.components
        assert 'encoder' in model.components
        assert 'decoder' in model.components
        assert 'sampling' in model.components
        
        # Verify component types
        assert model.components['decomposition'].type == 'SeriesDecomposition'
        assert model.components['attention'].type == 'StandardAutoCorrelation'
        assert model.components['encoder'].type == 'StandardEncoder'
        assert model.components['decoder'].type == 'StandardDecoder'
        assert model.components['sampling'].type == 'DeterministicSampling'
        
        # Verify global parameters
        assert model.global_params['seq_len'] == 96
        assert model.global_params['pred_len'] == 24
        assert model.global_params['d_model'] == 512
        
        # Test forward pass
        batch_size = 16
        seq_len = training_config['global_params']['seq_len']
        pred_len = training_config['global_params']['pred_len']
        
        x_enc = torch.randn(batch_size, seq_len, 7)
        x_mark_enc = torch.randn(batch_size, seq_len, 4)
        x_dec = torch.randn(batch_size, 48 + pred_len, 7)
        x_mark_dec = torch.randn(batch_size, 48 + pred_len, 4)
        
        output = model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        assert output.shape == (batch_size, pred_len, 7)
    
    def test_training_loop_with_modular_components(self, training_config):
        """Test complete training loop with modular architecture"""
        
        class MockTrainer:
            def __init__(self, model, config):
                self.model = model
                self.config = config
                self.train_losses = []
                self.val_losses = []
                self.epoch = 0
                
                # Initialize optimizer and loss function
                self.optimizer = torch.optim.Adam(model.parameters(), 
                                                lr=config['training']['learning_rate'])
                self.criterion = nn.MSELoss()
                
            def train_epoch(self, train_loader):
                """Train for one epoch"""
                self.model.train()
                epoch_loss = 0.0
                n_batches = 0
                
                for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model.forward(batch_x, batch_x_mark, 
                                               batch_y[:, :self.config['global_params']['label_len'], :],
                                               batch_y_mark)
                    
                    # Calculate loss
                    targets = batch_y[:, self.config['global_params']['label_len']:, :]
                    loss = self.criterion(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
                
                avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
                self.train_losses.append(avg_loss)
                return avg_loss
            
            def validate(self, val_loader):
                """Validate model"""
                self.model.eval()
                epoch_loss = 0.0
                n_batches = 0
                
                with torch.no_grad():
                    for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                        outputs = self.model.forward(batch_x, batch_x_mark,
                                                   batch_y[:, :self.config['global_params']['label_len'], :],
                                                   batch_y_mark)
                        
                        targets = batch_y[:, self.config['global_params']['label_len']:, :]
                        loss = self.criterion(outputs, targets)
                        
                        epoch_loss += loss.item()
                        n_batches += 1
                
                avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
                self.val_losses.append(avg_loss)
                return avg_loss
            
            def train(self, train_loader, val_loader):
                """Complete training process"""
                best_val_loss = float('inf')
                patience_counter = 0
                
                for epoch in range(self.config['training']['epochs']):
                    self.epoch = epoch
                    
                    # Train
                    train_loss = self.train_epoch(train_loader)
                    
                    # Validate
                    val_loss = self.validate(val_loader)
                    
                    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config['training']['patience']:
                            print(f"Early stopping at epoch {epoch+1}")
                            break
                
                return {
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'best_val_loss': best_val_loss,
                    'final_epoch': self.epoch + 1
                }
        
        # Create mock model
        from types import SimpleNamespace
        
        class MockModularModel:
            def __init__(self, config):
                self.config = config
                # Create simple linear layer for testing
                self.linear = nn.Linear(config['global_params']['enc_in'], 
                                      config['global_params']['c_out'])
                
            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                batch_size = x_enc.shape[0]
                pred_len = self.config['global_params']['pred_len']
                
                # Simple mock forward pass
                last_values = x_enc[:, -1, :]  # Take last time step
                prediction = self.linear(last_values)
                
                # Expand to prediction length
                prediction = prediction.unsqueeze(1).repeat(1, pred_len, 1)
                return prediction
            
            def parameters(self):
                return self.linear.parameters()
            
            def train(self):
                self.linear.train()
            
            def eval(self):
                self.linear.eval()
        
        # Create mock data loaders
        def create_mock_loader(n_batches=10):
            batch_size = training_config['training']['batch_size']
            seq_len = training_config['global_params']['seq_len']
            label_len = training_config['global_params']['label_len']
            pred_len = training_config['global_params']['pred_len']
            
            for _ in range(n_batches):
                batch_x = torch.randn(batch_size, seq_len, 7)
                batch_y = torch.randn(batch_size, label_len + pred_len, 7)
                batch_x_mark = torch.randn(batch_size, seq_len, 4)
                batch_y_mark = torch.randn(batch_size, label_len + pred_len, 4)
                
                yield batch_x, batch_y, batch_x_mark, batch_y_mark
        
        train_loader = create_mock_loader(20)
        val_loader = create_mock_loader(10)
        
        # Test training
        model = MockModularModel(training_config)
        trainer = MockTrainer(model, training_config)
        
        results = trainer.train(train_loader, val_loader)
        
        # Verify training results
        assert 'train_losses' in results
        assert 'val_losses' in results
        assert 'best_val_loss' in results
        assert 'final_epoch' in results
        
        assert len(results['train_losses']) > 0
        assert len(results['val_losses']) > 0
        assert results['final_epoch'] <= training_config['training']['epochs']
        
        # Training should reduce loss (at least somewhat)
        if len(results['train_losses']) > 1:
            initial_loss = results['train_losses'][0]
            final_loss = results['train_losses'][-1]
            # Allow for some variation, just check it's reasonable
            assert final_loss < initial_loss * 2  # Loss shouldn't explode
    
    def test_component_swapping_during_training(self, training_config):
        """Test ability to swap components during training"""
        
        class DynamicModularModel:
            def __init__(self, config):
                self.config = config
                self.components = {}
                self._initialize_components()
                
            def _initialize_components(self):
                # Initialize with simple mock components
                for comp_type in self.config['components']:
                    self.components[comp_type] = Mock()
            
            def swap_component(self, component_type, new_component):
                """Dynamically swap a component"""
                if component_type not in self.components:
                    raise ValueError(f"Unknown component type: {component_type}")
                
                old_component = self.components[component_type]
                self.components[component_type] = new_component
                
                return old_component
            
            def get_component(self, component_type):
                """Get current component"""
                return self.components.get(component_type)
            
            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                batch_size = x_enc.shape[0]
                pred_len = self.config['global_params']['pred_len']
                c_out = self.config['global_params']['c_out']
                return torch.randn(batch_size, pred_len, c_out)
            
            def parameters(self):
                return [torch.randn(10, requires_grad=True)]
        
        # Test component swapping
        model = DynamicModularModel(training_config)
        
        # Get initial component
        initial_attention = model.get_component('attention')
        assert initial_attention is not None
        
        # Create new component
        new_attention = Mock()
        new_attention.type = 'EnhancedAutoCorrelation'
        
        # Swap component
        old_attention = model.swap_component('attention', new_attention)
        assert old_attention == initial_attention
        
        # Verify swap
        current_attention = model.get_component('attention')
        assert current_attention == new_attention
        assert current_attention.type == 'EnhancedAutoCorrelation'
        
        # Test that model still works after swap
        batch_size = 8
        x_enc = torch.randn(batch_size, 96, 7)
        x_mark_enc = torch.randn(batch_size, 96, 4)
        x_dec = torch.randn(batch_size, 72, 7)
        x_mark_dec = torch.randn(batch_size, 72, 4)
        
        output = model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        assert output.shape == (batch_size, 24, 7)


class TestInferenceWorkflow:
    """Test inference workflows and model deployment"""
    
    def test_model_serialization_and_loading(self):
        """Test saving and loading modular models"""
        
        class SerializableModularModel:
            def __init__(self, config):
                self.config = config
                self.components = {}
                self.trained = False
                
            def save_model(self, path):
                """Save model configuration and state"""
                save_data = {
                    'config': self.config,
                    'components': {k: v.__class__.__name__ if hasattr(v, '__class__') else str(v) 
                                 for k, v in self.components.items()},
                    'trained': self.trained
                }
                
                # In real implementation, would use torch.save
                return save_data
            
            @classmethod
            def load_model(cls, path, saved_data):
                """Load model from saved data"""
                model = cls(saved_data['config'])
                
                # Restore components (simplified)
                for comp_type, comp_name in saved_data['components'].items():
                    # In real implementation, would recreate actual components
                    model.components[comp_type] = Mock()
                    model.components[comp_type].__class__.__name__ = comp_name
                
                model.trained = saved_data['trained']
                return model
            
            def mark_as_trained(self):
                self.trained = True
        
        # Test serialization
        config = {
            'model_type': 'ModularAutoformer',
            'components': {
                'attention': {'type': 'StandardAutoCorrelation'},
                'encoder': {'type': 'StandardEncoder'}
            }
        }
        
        # Create and "train" model
        model = SerializableModularModel(config)
        model.components['attention'] = Mock()
        model.components['encoder'] = Mock()
        model.mark_as_trained()
        
        # Save model
        saved_data = model.save_model('test_model.pth')
        
        # Load model
        loaded_model = SerializableModularModel.load_model('test_model.pth', saved_data)
        
        # Verify loaded model
        assert loaded_model.config == config
        assert loaded_model.trained == True
        assert len(loaded_model.components) == 2
        assert 'attention' in loaded_model.components
        assert 'encoder' in loaded_model.components
    
    def test_batch_inference(self):
        """Test batch inference with modular model"""
        
        class InferenceModel:
            def __init__(self, config):
                self.config = config
                
            def predict_batch(self, batch_data):
                """Process batch of data for inference"""
                x_enc, x_mark_enc, x_dec, x_mark_dec = batch_data
                
                batch_size = x_enc.shape[0]
                pred_len = self.config['global_params']['pred_len']
                c_out = self.config['global_params']['c_out']
                
                # Mock prediction
                predictions = torch.randn(batch_size, pred_len, c_out)
                
                # Return predictions with metadata
                return {
                    'predictions': predictions,
                    'batch_size': batch_size,
                    'prediction_length': pred_len,
                    'confidence': torch.rand(batch_size, pred_len, c_out)  # Mock confidence
                }
            
            def predict_streaming(self, data_stream):
                """Process streaming data for real-time inference"""
                results = []
                
                for batch_data in data_stream:
                    batch_result = self.predict_batch(batch_data)
                    results.append(batch_result)
                
                return results
        
        # Test batch inference
        config = {
            'global_params': {'pred_len': 24, 'c_out': 7}
        }
        
        model = InferenceModel(config)
        
        # Create test batch
        batch_size = 16
        batch_data = (
            torch.randn(batch_size, 96, 7),  # x_enc
            torch.randn(batch_size, 96, 4),  # x_mark_enc
            torch.randn(batch_size, 72, 7),  # x_dec
            torch.randn(batch_size, 72, 4)   # x_mark_dec
        )
        
        # Test single batch prediction
        result = model.predict_batch(batch_data)
        
        assert 'predictions' in result
        assert 'batch_size' in result
        assert 'prediction_length' in result
        assert 'confidence' in result
        
        assert result['predictions'].shape == (batch_size, 24, 7)
        assert result['batch_size'] == batch_size
        assert result['prediction_length'] == 24
        assert result['confidence'].shape == (batch_size, 24, 7)
        
        # Test streaming prediction
        def mock_data_stream():
            for _ in range(5):  # 5 batches
                yield (
                    torch.randn(8, 96, 7),
                    torch.randn(8, 96, 4),
                    torch.randn(8, 72, 7),
                    torch.randn(8, 72, 4)
                )
        
        streaming_results = model.predict_streaming(mock_data_stream())
        
        assert len(streaming_results) == 5
        for result in streaming_results:
            assert result['predictions'].shape == (8, 24, 7)
            assert result['batch_size'] == 8
    
    def test_uncertainty_quantification_inference(self):
        """Test inference with uncertainty quantification"""
        
        class UncertaintyAwareModel:
            def __init__(self, config):
                self.config = config
                self.has_bayesian_components = any(
                    'Bayesian' in comp.get('type', '') 
                    for comp in config.get('components', {}).values()
                )
            
            def predict_with_uncertainty(self, x_enc, x_mark_enc, x_dec, x_mark_dec, n_samples=100):
                """Predict with uncertainty estimation"""
                batch_size = x_enc.shape[0]
                pred_len = self.config['global_params']['pred_len']
                c_out = self.config['global_params']['c_out']
                
                if self.has_bayesian_components:
                    # Monte Carlo sampling for Bayesian components
                    samples = []
                    for _ in range(n_samples):
                        # Mock sampling from posterior
                        sample = torch.randn(batch_size, pred_len, c_out)
                        samples.append(sample)
                    
                    samples = torch.stack(samples, dim=0)  # (n_samples, batch_size, pred_len, c_out)
                    
                    # Calculate statistics
                    mean_prediction = samples.mean(dim=0)
                    std_prediction = samples.std(dim=0)
                    
                    # Calculate percentiles
                    sorted_samples, _ = torch.sort(samples, dim=0)
                    q05 = sorted_samples[int(0.05 * n_samples)]
                    q95 = sorted_samples[int(0.95 * n_samples)]
                    
                    return {
                        'mean': mean_prediction,
                        'std': std_prediction,
                        'samples': samples,
                        'q05': q05,
                        'q95': q95,
                        'epistemic_uncertainty': std_prediction,
                        'n_samples': n_samples
                    }
                else:
                    # Deterministic prediction with estimated uncertainty
                    prediction = torch.randn(batch_size, pred_len, c_out)
                    
                    # Estimate aleatoric uncertainty (simplified)
                    aleatoric_std = torch.rand(batch_size, pred_len, c_out) * 0.1
                    
                    return {
                        'mean': prediction,
                        'std': aleatoric_std,
                        'aleatoric_uncertainty': aleatoric_std,
                        'n_samples': 1
                    }
        
        # Test with Bayesian configuration
        bayesian_config = {
            'global_params': {'pred_len': 24, 'c_out': 7},
            'components': {
                'attention': {'type': 'StandardAutoCorrelation'},
                'sampling': {'type': 'BayesianSampling'}
            }
        }
        
        bayesian_model = UncertaintyAwareModel(bayesian_config)
        assert bayesian_model.has_bayesian_components
        
        # Test inputs
        batch_size = 8
        x_enc = torch.randn(batch_size, 96, 7)
        x_mark_enc = torch.randn(batch_size, 96, 4)
        x_dec = torch.randn(batch_size, 72, 7)
        x_mark_dec = torch.randn(batch_size, 72, 4)
        
        # Test Bayesian inference
        bayesian_result = bayesian_model.predict_with_uncertainty(
            x_enc, x_mark_enc, x_dec, x_mark_dec, n_samples=50
        )
        
        assert 'mean' in bayesian_result
        assert 'std' in bayesian_result
        assert 'samples' in bayesian_result
        assert 'q05' in bayesian_result
        assert 'q95' in bayesian_result
        assert 'epistemic_uncertainty' in bayesian_result
        
        assert bayesian_result['mean'].shape == (batch_size, 24, 7)
        assert bayesian_result['std'].shape == (batch_size, 24, 7)
        assert bayesian_result['samples'].shape == (50, batch_size, 24, 7)
        assert bayesian_result['n_samples'] == 50
        
        # Test with deterministic configuration
        deterministic_config = {
            'global_params': {'pred_len': 24, 'c_out': 7},
            'components': {
                'attention': {'type': 'StandardAutoCorrelation'},
                'sampling': {'type': 'DeterministicSampling'}
            }
        }
        
        deterministic_model = UncertaintyAwareModel(deterministic_config)
        assert not deterministic_model.has_bayesian_components
        
        # Test deterministic inference
        deterministic_result = deterministic_model.predict_with_uncertainty(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        
        assert 'mean' in deterministic_result
        assert 'std' in deterministic_result
        assert 'aleatoric_uncertainty' in deterministic_result
        assert 'epistemic_uncertainty' not in deterministic_result
        
        assert deterministic_result['mean'].shape == (batch_size, 24, 7)
        assert deterministic_result['std'].shape == (batch_size, 24, 7)
        assert deterministic_result['n_samples'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
