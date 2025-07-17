#!/usr/bin/env python3
"""
End-to-End Multi-Time Series Model Tests with Dimension Management

This comprehensive test suite addresses real-world dimension issues when running
models with multiple time series datasets that have different feature compositions.

Key Issues Addressed:
1. Dimension mismatches between different datasets
2. Dynamic model reconfiguration for varying feature sets
3. Batch processing with heterogeneous time series
4. Memory-efficient handling of large feature sets
5. Proper dimension flow from data loading to model output

Focus Areas:
- HFAutoformer models with varying backbone configurations
- DimensionManager integration for automatic dimension handling
- Real-world data scenarios (financial, weather, energy, etc.)
- Performance optimization with large-scale multi-variate time series
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from utils.dimension_manager import DimensionManager


@dataclass
class DatasetSpec:
    """Specification for a test dataset"""
    name: str
    target_features: List[str]
    all_features: List[str]
    n_series: int
    seq_len: int
    pred_len: int
    frequency: str = 'daily'
    
    @property
    def n_targets(self) -> int:
        return len(self.target_features)
    
    @property
    def n_covariates(self) -> int:
        return len(self.all_features) - len(self.target_features)
    
    @property
    def n_total_features(self) -> int:
        return len(self.all_features)


class MockHFAutoformer(nn.Module):
    """
    Mock HFAutoformer for testing dimension handling
    Simulates the key dimension transformation points in the real model
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.d_model = config.get('d_model', 512)
        
        # Use DimensionManager for automatic dimension handling
        self.dim_manager = DimensionManager(
            mode=config['mode'],
            target_features=config['target_features'],
            all_features=config['all_features'],
            loss_function=config.get('loss_function', 'mse'),
            quantiles=config.get('quantiles', [])
        )
        
        # Model components with dimension-aware initialization
        self.input_embedding = nn.Linear(self.dim_manager.enc_in, self.d_model)
        self.decoder_input_proj = nn.Linear(self.dim_manager.dec_in, self.d_model)
        
        # Mock transformer backbone (simplified)
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=8,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Output projection with proper quantile scaling
        self.output_projection = nn.Linear(self.d_model, self.dim_manager.c_out_model)
        
        # Store dimension info for validation
        self.enc_in = self.dim_manager.enc_in
        self.dec_in = self.dim_manager.dec_in
        self.c_out_model = self.dim_manager.c_out_model
        self.c_out_evaluation = self.dim_manager.c_out_evaluation
        
        print(f"  🏗️ MockHFAutoformer initialized:")
        print(f"     Mode: {config['mode']}")
        print(f"     Input dims: {self.enc_in} (enc), {self.dec_in} (dec)")
        print(f"     Output dims: {self.c_out_model} (model), {self.c_out_evaluation} (eval)")
        print(f"     Dimension Manager: {self.dim_manager}")
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward pass with dimension validation at each step
        """
        batch_size, seq_len, _ = x_enc.shape
        _, dec_len, _ = x_dec.shape
        
        # Validate input dimensions
        assert x_enc.shape[-1] == self.enc_in, f"Encoder input dim mismatch: got {x_enc.shape[-1]}, expected {self.enc_in}"
        assert x_dec.shape[-1] == self.dec_in, f"Decoder input dim mismatch: got {x_dec.shape[-1]}, expected {self.dec_in}"
        
        # Embedding layers
        enc_embedded = self.input_embedding(x_enc)  # [batch, seq_len, d_model]
        dec_embedded = self.decoder_input_proj(x_dec)  # [batch, dec_len, d_model]
        
        # Mock backbone processing
        backbone_output = self.backbone(enc_embedded)  # [batch, seq_len, d_model]
        
        # Extract prediction portion (last pred_len steps)
        pred_len = dec_len - (seq_len // 2)  # Assuming label_len = seq_len // 2
        pred_portion = backbone_output[:, -pred_len:, :]  # [batch, pred_len, d_model]
        
        # Final output projection
        output = self.output_projection(pred_portion)  # [batch, pred_len, c_out_model]
        
        # Validate output dimensions
        assert output.shape[-1] == self.c_out_model, f"Output dim mismatch: got {output.shape[-1]}, expected {self.c_out_model}"
        
        return output
    
    def get_dimension_info(self) -> Dict:
        """Get dimension information for debugging"""
        return {
            'enc_in': self.enc_in,
            'dec_in': self.dec_in,
            'c_out_model': self.c_out_model,
            'c_out_evaluation': self.c_out_evaluation,
            'mode': self.config['mode'],
            'targets': len(self.config['target_features']),
            'all_features': len(self.config['all_features'])
        }


class TestEndToEndMultiTimeSeries:
    """
    Comprehensive end-to-end tests for multi-time series dimension handling
    """
    
    def setup_method(self):
        """Setup test datasets and configurations"""
        
        # Define diverse test datasets
        self.datasets = {
            'financial': DatasetSpec(
                name='financial',
                target_features=['open', 'high', 'low', 'close', 'volume'],
                all_features=['open', 'high', 'low', 'close', 'volume', 'market_cap', 'pe_ratio', 'rsi', 'macd'],
                n_series=100,
                seq_len=96,
                pred_len=24
            ),
            'weather': DatasetSpec(
                name='weather',
                target_features=['temperature', 'humidity'],
                all_features=['temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover'],
                n_series=50,
                seq_len=168,  # 1 week hourly
                pred_len=24   # 1 day ahead
            ),
            'energy': DatasetSpec(
                name='energy',
                target_features=['consumption'],
                all_features=['consumption', 'temperature', 'hour', 'day_of_week', 'is_holiday'],
                n_series=200,
                seq_len=144,  # 6 days 15-min intervals
                pred_len=96   # 1 day ahead
            ),
            'minimal': DatasetSpec(
                name='minimal',
                target_features=['value'],
                all_features=['value'],
                n_series=10,
                seq_len=48,
                pred_len=12
            ),
            'large_multivariate': DatasetSpec(
                name='large_multivariate',
                target_features=[f'target_{i}' for i in range(20)],
                all_features=[f'target_{i}' for i in range(20)] + [f'cov_{i}' for i in range(80)],
                n_series=50,
                seq_len=96,
                pred_len=24
            )
        }
        
        print("🧪 End-to-End Multi-Time Series Test Suite")
        print("=" * 70)
        print("📊 Test Datasets:")
        for name, spec in self.datasets.items():
            print(f"  - {name}: {spec.n_targets} targets, {spec.n_covariates} covariates, {spec.n_series} series")
    
    def generate_synthetic_data(self, dataset_spec: DatasetSpec, batch_size: int = 8) -> Tuple[torch.Tensor, ...]:
        """Generate synthetic data matching dataset specification"""
        
        # Input sequences
        x_enc = torch.randn(batch_size, dataset_spec.seq_len, dataset_spec.n_total_features)
        
        # Decoder sequences (label_len + pred_len)
        label_len = dataset_spec.seq_len // 2
        total_dec_len = label_len + dataset_spec.pred_len
        x_dec = torch.randn(batch_size, total_dec_len, dataset_spec.n_total_features)
        
        # Time markings (simplified - day of week, day of month, month)
        n_time_features = 3
        x_mark_enc = torch.randn(batch_size, dataset_spec.seq_len, n_time_features)
        x_mark_dec = torch.randn(batch_size, total_dec_len, n_time_features)
        
        return x_enc, x_mark_enc, x_dec, x_mark_dec
    
    def test_single_dataset_dimension_flow(self):
        """Test dimension flow for individual datasets"""
        print("\n🔄 Testing Single Dataset Dimension Flow")
        
        for dataset_name, dataset_spec in self.datasets.items():
            print(f"\n  📈 Testing {dataset_name} dataset:")
            
            for mode in ['S', 'MS', 'M']:
                # Create model configuration
                config = {
                    'mode': mode,
                    'target_features': dataset_spec.target_features,
                    'all_features': dataset_spec.all_features,
                    'loss_function': 'mse',
                    'd_model': 256
                }
                
                # Initialize model
                model = MockHFAutoformer(config)
                
                # Generate test data
                x_enc, x_mark_enc, x_dec, x_mark_dec = self.generate_synthetic_data(dataset_spec)
                
                # Forward pass
                try:
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    
                    # Validate output shape
                    expected_shape = (x_enc.shape[0], dataset_spec.pred_len, model.c_out_model)
                    assert output.shape == expected_shape, f"Output shape mismatch: got {output.shape}, expected {expected_shape}"
                    
                    print(f"    ✅ {mode} mode: {model.enc_in}→{model.c_out_model} | Output: {output.shape}")
                    
                except Exception as e:
                    print(f"    ❌ {mode} mode failed: {e}")
                    raise
    
    def test_multi_dataset_batch_processing(self):
        """Test batch processing across different datasets"""
        print("\n🔄 Testing Multi-Dataset Batch Processing")
        
        mode = 'MS'  # Focus on multivariate-to-multivariate for complex scenarios
        
        for dataset_name, dataset_spec in self.datasets.items():
            print(f"\n  🔢 Testing batch processing for {dataset_name}:")
            
            config = {
                'mode': mode,
                'target_features': dataset_spec.target_features,
                'all_features': dataset_spec.all_features,
                'loss_function': 'mse',
                'd_model': 256
            }
            
            model = MockHFAutoformer(config)
            
            # Test different batch sizes
            batch_sizes = [1, 4, 8, 16, 32]
            
            for batch_size in batch_sizes:
                try:
                    x_enc, x_mark_enc, x_dec, x_mark_dec = self.generate_synthetic_data(
                        dataset_spec, batch_size=batch_size
                    )
                    
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    
                    expected_shape = (batch_size, dataset_spec.pred_len, model.c_out_model)
                    assert output.shape == expected_shape
                    
                    print(f"    ✅ Batch size {batch_size}: {output.shape}")
                    
                except Exception as e:
                    print(f"    ❌ Batch size {batch_size} failed: {e}")
                    raise
    
    def test_quantile_dimension_handling(self):
        """Test dimension handling with quantile losses"""
        print("\n📊 Testing Quantile Dimension Handling")
        
        quantile_configs = [
            ("3-quantile", [0.1, 0.5, 0.9]),
            ("5-quantile", [0.05, 0.25, 0.5, 0.75, 0.95]),
            ("9-quantile", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        ]
        
        # Test with financial dataset (complex multivariate scenario)
        dataset_spec = self.datasets['financial']
        
        for quantile_name, quantiles in quantile_configs:
            print(f"\n  📈 Testing {quantile_name} configuration:")
            
            for mode in ['S', 'MS', 'M']:
                config = {
                    'mode': mode,
                    'target_features': dataset_spec.target_features,
                    'all_features': dataset_spec.all_features,
                    'loss_function': 'quantile',
                    'quantiles': quantiles,
                    'd_model': 256
                }
                
                model = MockHFAutoformer(config)
                
                # Generate test data
                x_enc, x_mark_enc, x_dec, x_mark_dec = self.generate_synthetic_data(dataset_spec)
                
                try:
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    
                    # Validate quantile scaling
                    base_output_dim = model.c_out_evaluation
                    expected_model_dim = base_output_dim * len(quantiles)
                    
                    assert model.c_out_model == expected_model_dim
                    assert output.shape[-1] == expected_model_dim
                    
                    # Test quantile reshaping for evaluation
                    batch_size, pred_len, _ = output.shape
                    reshaped = output.reshape(batch_size, pred_len, base_output_dim, len(quantiles))
                    
                    assert reshaped.shape == (batch_size, pred_len, base_output_dim, len(quantiles))
                    
                    print(f"    ✅ {mode} mode: {base_output_dim}×{len(quantiles)}={expected_model_dim} | "
                          f"Reshaped: {reshaped.shape}")
                    
                except Exception as e:
                    print(f"    ❌ {mode} mode with {quantile_name} failed: {e}")
                    raise
    
    def test_dynamic_dataset_switching(self):
        """Test switching between datasets with different dimensions"""
        print("\n🔄 Testing Dynamic Dataset Switching")
        
        # Simulate scenario where model needs to handle different datasets
        dataset_sequence = ['minimal', 'weather', 'financial', 'energy', 'large_multivariate']
        mode = 'MS'
        
        models = {}
        
        for dataset_name in dataset_sequence:
            dataset_spec = self.datasets[dataset_name]
            print(f"\n  🔧 Configuring for {dataset_name} dataset:")
            
            config = {
                'mode': mode,
                'target_features': dataset_spec.target_features,
                'all_features': dataset_spec.all_features,
                'loss_function': 'mse',
                'd_model': 256
            }
            
            # Create model for this dataset
            model = MockHFAutoformer(config)
            models[dataset_name] = model
            
            # Test with dataset
            x_enc, x_mark_enc, x_dec, x_mark_dec = self.generate_synthetic_data(dataset_spec)
            
            try:
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                dim_info = model.get_dimension_info()
                print(f"    ✅ {dataset_name}: {dim_info['enc_in']}→{dim_info['c_out_model']} | "
                      f"Targets: {dim_info['targets']}, Features: {dim_info['all_features']}")
                
            except Exception as e:
                print(f"    ❌ {dataset_name} failed: {e}")
                raise
        
        # Verify models have different dimensions
        dim_infos = {name: model.get_dimension_info() for name, model in models.items()}
        
        # Check that models properly adapt to different feature sets
        assert dim_infos['minimal']['enc_in'] != dim_infos['financial']['enc_in']
        assert dim_infos['weather']['targets'] != dim_infos['energy']['targets']
        assert dim_infos['large_multivariate']['all_features'] > dim_infos['minimal']['all_features']
        
        print("\n  ✅ All models properly configured with different dimensions")
    
    def test_memory_scaling_analysis(self):
        """Test memory scaling with different dataset sizes"""
        print("\n💾 Testing Memory Scaling Analysis")
        
        mode = 'M'  # Most memory-intensive mode
        batch_size = 16
        
        for dataset_name, dataset_spec in self.datasets.items():
            print(f"\n  📊 Memory analysis for {dataset_name}:")
            
            config = {
                'mode': mode,
                'target_features': dataset_spec.target_features,
                'all_features': dataset_spec.all_features,
                'loss_function': 'mse',
                'd_model': 512  # Larger model for memory testing
            }
            
            model = MockHFAutoformer(config)
            
            # Generate data
            x_enc, x_mark_enc, x_dec, x_mark_dec = self.generate_synthetic_data(
                dataset_spec, batch_size=batch_size
            )
            
            # Estimate memory footprint
            input_elements = x_enc.numel() + x_dec.numel()
            
            try:
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                output_elements = output.numel()
                
                total_elements = input_elements + output_elements
                memory_mb = total_elements * 4 / (1024 * 1024)  # 4 bytes per float32
                
                print(f"    📈 Input: {x_enc.shape}, Output: {output.shape}")
                print(f"    💾 Memory: ~{memory_mb:.2f} MB")
                print(f"    🔢 Features: {dataset_spec.n_total_features}, Targets: {dataset_spec.n_targets}")
                
                # Basic memory sanity checks
                assert memory_mb < 1000, f"Memory usage too high: {memory_mb:.2f} MB"
                assert output_elements > 0
                
                print(f"    ✅ Memory scaling acceptable")
                
            except Exception as e:
                print(f"    ❌ Memory test failed: {e}")
                raise
    
    def test_error_handling_and_recovery(self):
        """Test error handling for dimension mismatches and recovery"""
        print("\n⚠️ Testing Error Handling and Recovery")
        
        dataset_spec = self.datasets['financial']
        
        # Test 1: Wrong input dimensions
        print("\n  🔍 Testing wrong input dimensions:")
        
        config = {
            'mode': 'MS',
            'target_features': dataset_spec.target_features,
            'all_features': dataset_spec.all_features,
            'loss_function': 'mse',
            'd_model': 256
        }
        
        model = MockHFAutoformer(config)
        
        # Generate data with wrong dimensions
        wrong_features = dataset_spec.n_total_features + 5  # Wrong number of features
        x_enc_wrong = torch.randn(8, dataset_spec.seq_len, wrong_features)
        x_mark_enc = torch.randn(8, dataset_spec.seq_len, 3)
        x_dec_wrong = torch.randn(8, 60, wrong_features)
        x_mark_dec = torch.randn(8, 60, 3)
        
        try:
            output = model(x_enc_wrong, x_mark_enc, x_dec_wrong, x_mark_dec)
            print("    ❌ Should have failed with dimension mismatch")
            assert False, "Expected dimension mismatch error"
        except AssertionError as e:
            if "dim mismatch" in str(e):
                print("    ✅ Correctly detected input dimension mismatch")
            else:
                raise
        
        # Test 2: Recovery with correct dimensions
        print("\n  🔧 Testing recovery with correct dimensions:")
        
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.generate_synthetic_data(dataset_spec)
        
        try:
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            print("    ✅ Successfully recovered with correct dimensions")
        except Exception as e:
            print(f"    ❌ Recovery failed: {e}")
            raise
    
    def test_cross_dataset_compatibility(self):
        """Test compatibility when models encounter data from different datasets"""
        print("\n🔄 Testing Cross-Dataset Compatibility")
        
        # Create model for financial dataset
        financial_spec = self.datasets['financial']
        config = {
            'mode': 'MS',
            'target_features': financial_spec.target_features,
            'all_features': financial_spec.all_features,
            'loss_function': 'mse',
            'd_model': 256
        }
        
        financial_model = MockHFAutoformer(config)
        
        print(f"  🏦 Financial model: {financial_model.enc_in} features")
        
        # Test with compatible data (same feature count)
        print("\n  ✅ Testing with compatible data:")
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.generate_synthetic_data(financial_spec)
        
        try:
            output = financial_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            print(f"    ✅ Compatible data works: {output.shape}")
        except Exception as e:
            print(f"    ❌ Compatible data failed: {e}")
            raise
        
        # Test with incompatible data (different feature count)
        print("\n  ❌ Testing with incompatible data:")
        weather_spec = self.datasets['weather']
        x_enc_wrong, x_mark_enc, x_dec_wrong, x_mark_dec = self.generate_synthetic_data(weather_spec)
        
        try:
            output = financial_model(x_enc_wrong, x_mark_enc, x_dec_wrong, x_mark_dec)
            print("    ❌ Should have failed with incompatible data")
            assert False, "Expected incompatibility error"
        except AssertionError as e:
            if "dim mismatch" in str(e):
                print(f"    ✅ Correctly detected incompatible data: {weather_spec.n_total_features} vs {financial_spec.n_total_features} features")
            else:
                raise


def run_end_to_end_tests():
    """Run all end-to-end multi-time series tests"""
    print("🚀 Running End-to-End Multi-Time Series Test Suite")
    print("=" * 80)
    
    test_instance = TestEndToEndMultiTimeSeries()
    test_instance.setup_method()
    
    try:
        test_instance.test_single_dataset_dimension_flow()
        test_instance.test_multi_dataset_batch_processing()
        test_instance.test_quantile_dimension_handling()
        test_instance.test_dynamic_dataset_switching()
        test_instance.test_memory_scaling_analysis()
        test_instance.test_error_handling_and_recovery()
        test_instance.test_cross_dataset_compatibility()
        
        print("\n🎉 🎉 🎉 ALL END-TO-END TESTS PASSED! 🎉 🎉 🎉")
        print("=" * 80)
        print("\n✅ Key Achievements:")
        print("  - Multi-dataset dimension handling validated")
        print("  - DimensionManager integration confirmed")
        print("  - Quantile scaling properly implemented")
        print("  - Error detection and recovery working")
        print("  - Memory scaling acceptable across all scenarios")
        print("  - Cross-dataset compatibility correctly enforced")
        
    except Exception as e:
        print(f"\n❌ End-to-End Tests Failed: {e}")
        raise


if __name__ == "__main__":
    run_end_to_end_tests()
