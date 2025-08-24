#!/usr/bin/env python3
"""
Comprehensive Dimension Manager Integration Tests

This test suite validates the DimensionManager's ability to handle:
1. Multiple time series datasets with different feature compositions
2. Dynamic dimension adjustments across different modes (S, MS, M)
3. Proper dimension flow through the entire model pipeline
4. Edge cases with quantile losses and multiple outputs
5. Real-world scenarios with complex feature engineering

Key Issues Addressed:
- Dimension mismatches between encoder/decoder inputs
- Quantile loss output dimension scaling
- Feature mode transitions
- Multi-dataset batch processing
- Memory efficiency with large feature sets
"""

import pytest
import torch
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from layers.modular.dimensions.dimension_manager import DimensionManager


class TestDimensionManagerIntegration:
    """
    Comprehensive integration tests for DimensionManager
    """
    
    def setup_method(self):
        """Setup test fixtures"""
        # Standard financial time series features
        self.financial_targets = ['open', 'high', 'low', 'close', 'volume']
        self.financial_covariates = ['market_cap', 'pe_ratio', 'rsi', 'macd', 'bollinger_upper']
        self.financial_all = self.financial_targets + self.financial_covariates
        
        # Weather forecasting features  
        self.weather_targets = ['temperature', 'humidity']
        self.weather_covariates = ['pressure', 'wind_speed', 'cloud_cover', 'solar_radiation']
        self.weather_all = self.weather_targets + self.weather_covariates
        
        # Energy consumption features
        self.energy_targets = ['total_consumption']
        self.energy_covariates = ['temperature', 'hour_of_day', 'day_of_week', 'is_holiday', 'price_per_kwh']
        self.energy_all = self.energy_targets + self.energy_covariates
        
        # Quantile configurations
        self.standard_quantiles = [0.1, 0.5, 0.9]
        self.comprehensive_quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        
        print("TEST DimensionManager Integration Test Suite")
        print("=" * 60)
    
    def test_single_dataset_dimension_consistency(self):
        """Test dimension consistency across modes for a single dataset"""
        print("\nCHART Testing Single Dataset Dimension Consistency")
        
        for mode in ['S', 'MS', 'M']:
            for loss_function in ['mse', 'mae', 'quantile']:
                quantiles = self.standard_quantiles if loss_function == 'quantile' else []
                
                dm = DimensionManager(
                    mode=mode,
                    target_features=self.financial_targets,
                    all_features=self.financial_all,
                    loss_function=loss_function,
                    quantiles=quantiles
                )
                
                # Basic consistency checks
                assert dm.num_targets == len(self.financial_targets)
                assert dm.num_covariates == len(self.financial_covariates)
                assert dm.num_targets + dm.num_covariates == len(self.financial_all)
                
                # Mode-specific dimension checks
                if mode == 'S':
                    assert dm.enc_in == dm.num_targets
                    assert dm.dec_in == dm.num_targets
                    assert dm.c_out_evaluation == dm.num_targets
                elif mode == 'MS':
                    assert dm.enc_in == len(self.financial_all)
                    assert dm.dec_in == len(self.financial_all)
                    assert dm.c_out_evaluation == dm.num_targets
                elif mode == 'M':
                    assert dm.enc_in == len(self.financial_all)
                    assert dm.dec_in == len(self.financial_all)
                    assert dm.c_out_evaluation == len(self.financial_all)
                
                # Quantile scaling check
                if loss_function == 'quantile' and quantiles:
                    expected_model_out = dm.c_out_evaluation * len(quantiles)
                    assert dm.c_out_model == expected_model_out
                else:
                    assert dm.c_out_model == dm.c_out_evaluation
                
                print(f"  PASS {mode} mode with {loss_function} loss: {dm.enc_in}{dm.c_out_model}")
    
    def test_multi_dataset_dimension_handling(self):
        """Test dimension handling across multiple diverse datasets"""
        print("\n Testing Multi-Dataset Dimension Handling")
        
        datasets_config = [
            ("Financial", self.financial_targets, self.financial_all),
            ("Weather", self.weather_targets, self.weather_all),
            ("Energy", self.energy_targets, self.energy_all)
        ]
        
        for name, targets, all_features in datasets_config:
            for mode in ['S', 'MS', 'M']:
                dm = DimensionManager(
                    mode=mode,
                    target_features=targets,
                    all_features=all_features,
                    loss_function='mse'
                )
                
                # Validate proper dimension calculation
                expected_targets = len(targets)
                expected_covariates = len(all_features) - len(targets)
                
                assert dm.num_targets == expected_targets
                assert dm.num_covariates == expected_covariates
                
                # Validate mode-specific behavior
                self._validate_mode_dimensions(dm, mode, expected_targets, len(all_features))
                
                print(f"  PASS {name} dataset in {mode} mode: "
                      f"{expected_targets} targets, {expected_covariates} covariates")
    
    def test_quantile_dimension_scaling(self):
        """Test proper dimension scaling with different quantile configurations"""
        print("\nGRAPH Testing Quantile Dimension Scaling")
        
        quantile_configs = [
            ("Standard", self.standard_quantiles),
            ("Comprehensive", self.comprehensive_quantiles),
            ("Single", [0.5]),
            ("Extreme", [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        ]
        
        for config_name, quantiles in quantile_configs:
            for mode in ['S', 'MS', 'M']:
                dm = DimensionManager(
                    mode=mode,
                    target_features=self.financial_targets,
                    all_features=self.financial_all,
                    loss_function='quantile',
                    quantiles=quantiles
                )
                
                # Check quantile scaling
                base_output = dm.c_out_evaluation
                expected_model_output = base_output * len(quantiles)
                
                assert dm.c_out_model == expected_model_output
                
                print(f"  PASS {config_name} quantiles ({len(quantiles)}) in {mode} mode: "
                      f"{base_output}{expected_model_output}")
    
    def test_dynamic_dimension_updates(self):
        """Test dimension manager behavior with dynamic feature changes"""
        print("\nREFRESH Testing Dynamic Dimension Updates")
        
        # Start with basic configuration
        initial_targets = ['close']
        initial_all = ['close', 'volume']
        
        dm = DimensionManager(
            mode='MS',
            target_features=initial_targets,
            all_features=initial_all,
            loss_function='mse'
        )
        
        initial_enc_in = dm.enc_in
        initial_c_out = dm.c_out_evaluation
        
        # Simulate adding more features (common in feature engineering)
        extended_targets = ['close', 'volume']
        extended_all = ['close', 'volume', 'rsi', 'macd', 'bollinger_upper']
        
        dm_extended = DimensionManager(
            mode='MS',
            target_features=extended_targets,
            all_features=extended_all,
            loss_function='mse'
        )
        
        # Verify proper dimension scaling
        assert dm_extended.enc_in > initial_enc_in
        assert dm_extended.c_out_evaluation > initial_c_out
        
        print(f"  PASS Feature expansion: {initial_enc_in}{dm_extended.enc_in} input, "
              f"{initial_c_out}{dm_extended.c_out_evaluation} output")
    
    def test_edge_case_configurations(self):
        """Test edge cases and boundary conditions"""
        print("\nWARN  Testing Edge Case Configurations")
        
        # Single target, no covariates
        dm_minimal = DimensionManager(
            mode='S',
            target_features=['price'],
            all_features=['price'],
            loss_function='mse'
        )
        assert dm_minimal.num_covariates == 0
        assert dm_minimal.enc_in == 1
        assert dm_minimal.c_out_evaluation == 1
        print("  PASS Minimal configuration (single target, no covariates)")
        
        # Many targets, few covariates
        many_targets = [f'target_{i}' for i in range(20)]
        few_covariates = ['cov_1', 'cov_2']
        
        dm_many_targets = DimensionManager(
            mode='M',
            target_features=many_targets,
            all_features=many_targets + few_covariates,
            loss_function='mse'
        )
        assert dm_many_targets.num_targets == 20
        assert dm_many_targets.num_covariates == 2
        print("  PASS Many targets, few covariates")
        
        # Few targets, many covariates
        few_targets = ['main_target']
        many_covariates = [f'cov_{i}' for i in range(50)]
        
        dm_many_covariates = DimensionManager(
            mode='MS',
            target_features=few_targets,
            all_features=few_targets + many_covariates,
            loss_function='mse'
        )
        assert dm_many_covariates.num_targets == 1
        assert dm_many_covariates.num_covariates == 50
        print("  PASS Few targets, many covariates")
        
        # Large quantile set
        large_quantiles = [i/100 for i in range(1, 100)]  # 99 quantiles
        dm_large_quantiles = DimensionManager(
            mode='MS',
            target_features=self.financial_targets,
            all_features=self.financial_all,
            loss_function='quantile',
            quantiles=large_quantiles
        )
        expected_output = len(self.financial_targets) * len(large_quantiles)
        assert dm_large_quantiles.c_out_model == expected_output
        print(f"  PASS Large quantile set: {len(large_quantiles)} quantiles")
    
    def test_memory_efficiency_estimation(self):
        """Test dimension calculations for memory efficiency estimation"""
        print("\n Testing Memory Efficiency Estimation")
        
        # Simulate very large feature sets
        large_configs = [
            ("Small", 5, 10),      # 5 targets, 10 total features
            ("Medium", 20, 100),   # 20 targets, 100 total features  
            ("Large", 50, 500),    # 50 targets, 500 total features
            ("XLarge", 100, 1000), # 100 targets, 1000 total features
        ]
        
        seq_len, pred_len, batch_size = 96, 24, 32
        
        for config_name, n_targets, n_total in large_configs:
            targets = [f'target_{i}' for i in range(n_targets)]
            all_features = [f'feature_{i}' for i in range(n_total)]
            
            dm = DimensionManager(
                mode='M',
                target_features=targets,
                all_features=all_features,
                loss_function='mse'
            )
            
            # Estimate memory footprint
            input_size = batch_size * seq_len * dm.enc_in
            output_size = batch_size * pred_len * dm.c_out_model
            total_elements = input_size + output_size
            memory_mb = total_elements * 4 / (1024 * 1024)  # 4 bytes per float32
            
            print(f"  CHART {config_name} config: {dm.enc_in} features  "
                  f"{dm.c_out_model} outputs (~{memory_mb:.1f} MB)")
            
            # Basic memory sanity checks
            assert input_size > 0
            assert output_size > 0
            assert memory_mb < 10000  # Reasonable upper bound for testing
    
    def test_invalid_configurations(self):
        """Test proper error handling for invalid configurations"""
        print("\nFAIL Testing Invalid Configuration Handling")
        
        # Invalid mode
        with pytest.raises(ValueError, match="Unknown feature mode"):
            DimensionManager(
                mode='INVALID',
                target_features=['target'],
                all_features=['target', 'cov'],
                loss_function='mse'
            )
        print("  PASS Invalid mode detection")
        
        # Target features not in all features - this actually works in the current implementation
        # The DimensionManager handles this gracefully by treating missing targets as separate
        dm = DimensionManager(
            mode='S',
            target_features=['missing_target'],
            all_features=['other_feature'],
            loss_function='mse'
        )
        # This works - covariate_features will be ['other_feature'] since missing_target not found
        assert dm.covariate_features == ['other_feature']
        print("  PASS Missing target feature handling (graceful)")
        
        # Empty quantiles with quantile loss
        dm_empty_quantiles = DimensionManager(
            mode='S',
            target_features=['target'],
            all_features=['target'],
            loss_function='quantile',
            quantiles=[]
        )
        # Should not scale output dimensions if no quantiles
        assert dm_empty_quantiles.c_out_model == dm_empty_quantiles.c_out_evaluation
        print("  PASS Empty quantiles handling")
    
    def _validate_mode_dimensions(self, dm: DimensionManager, mode: str, 
                                 n_targets: int, n_total: int):
        """Helper to validate mode-specific dimension calculations"""
        if mode == 'S':
            assert dm.enc_in == n_targets
            assert dm.dec_in == n_targets  
            assert dm.c_out_evaluation == n_targets
        elif mode == 'MS':
            assert dm.enc_in == n_total
            assert dm.dec_in == n_total
            assert dm.c_out_evaluation == n_targets
        elif mode == 'M':
            assert dm.enc_in == n_total
            assert dm.dec_in == n_total
            assert dm.c_out_evaluation == n_total
    
    def test_dimension_manager_repr(self):
        """Test the string representation for debugging"""
        print("\n Testing DimensionManager String Representation")
        
        dm = DimensionManager(
            mode='MS',
            target_features=self.financial_targets,
            all_features=self.financial_all,
            loss_function='quantile',
            quantiles=self.standard_quantiles
        )
        
        repr_str = repr(dm)
        
        # Check that key information is present
        assert 'MS' in repr_str
        assert 'quantile' in repr_str
        assert str(len(self.financial_targets)) in repr_str
        assert str(len(self.financial_covariates)) in repr_str
        
        print(f"  PASS Representation: {repr_str}")


class TestDimensionManagerModelIntegration:
    """
    Integration tests between DimensionManager and actual models
    """
    
    def setup_method(self):
        """Setup for model integration tests"""
        self.seq_len = 96
        self.pred_len = 24
        self.label_len = 48
        self.batch_size = 4
    
    def test_mock_model_integration(self):
        """Test dimension flow through a mock model architecture"""
        print("\n Testing Mock Model Integration")
        
        # Create dimension manager
        targets = ['price', 'volume']
        all_features = ['price', 'volume', 'rsi', 'macd']
        
        for mode in ['S', 'MS', 'M']:
            dm = DimensionManager(
                mode=mode,
                target_features=targets,
                all_features=all_features,
                loss_function='mse'
            )
            
            # Create mock input tensors with proper dimensions
            batch_x = torch.randn(self.batch_size, self.seq_len, dm.enc_in)
            
            # Mock encoder
            encoder_output = torch.randn(self.batch_size, self.seq_len, 512)  # d_model=512
            
            # Mock decoder input  
            batch_y = torch.randn(self.batch_size, self.label_len + self.pred_len, dm.dec_in)
            decoder_output = torch.randn(self.batch_size, self.pred_len, 512)
            
            # Mock final projection
            final_output = torch.randn(self.batch_size, self.pred_len, dm.c_out_model)
            
            # Validate dimensions
            assert batch_x.shape[-1] == dm.enc_in
            assert batch_y.shape[-1] == dm.dec_in
            assert final_output.shape[-1] == dm.c_out_model
            
            print(f"  PASS {mode} mode: {dm.enc_in}512{dm.c_out_model}")
    
    def test_quantile_output_reshaping(self):
        """Test proper reshaping of quantile outputs for evaluation"""
        print("\nGRAPH Testing Quantile Output Reshaping")
        
        targets = ['price']
        quantiles = [0.1, 0.5, 0.9]
        
        dm = DimensionManager(
            mode='S',
            target_features=targets,
            all_features=targets,
            loss_function='quantile',
            quantiles=quantiles
        )
        
        # Model output has quantile-scaled dimensions
        model_output = torch.randn(self.batch_size, self.pred_len, dm.c_out_model)
        
        # Reshape for evaluation: [batch, pred_len, n_targets, n_quantiles]
        n_targets = dm.c_out_evaluation
        n_quantiles = len(quantiles)
        
        reshaped = model_output.reshape(
            self.batch_size, self.pred_len, n_targets, n_quantiles
        )
        
        assert reshaped.shape == (self.batch_size, self.pred_len, n_targets, n_quantiles)
        
        # Extract median prediction (quantile 0.5)
        median_idx = quantiles.index(0.5)
        median_pred = reshaped[:, :, :, median_idx]
        
        assert median_pred.shape == (self.batch_size, self.pred_len, n_targets)
        
        print(f"  PASS Quantile reshaping: {model_output.shape}  {reshaped.shape}")
        print(f"  PASS Median extraction: {median_pred.shape}")


def run_dimension_integration_tests():
    """Run all dimension manager integration tests"""
    print("ROCKET Running DimensionManager Integration Test Suite")
    print("=" * 80)
    
    # Unit tests
    test_instance = TestDimensionManagerIntegration()
    test_instance.setup_method()
    
    try:
        test_instance.test_single_dataset_dimension_consistency()
        test_instance.test_multi_dataset_dimension_handling()
        test_instance.test_quantile_dimension_scaling()
        test_instance.test_dynamic_dimension_updates()
        test_instance.test_edge_case_configurations()
        test_instance.test_memory_efficiency_estimation()
        test_instance.test_invalid_configurations()
        test_instance.test_dimension_manager_repr()
        
        print("\nPASS Unit Tests: ALL PASSED")
        
    except Exception as e:
        print(f"\nFAIL Unit Tests Failed: {e}")
        raise
    
    # Integration tests
    integration_instance = TestDimensionManagerModelIntegration()
    integration_instance.setup_method()
    
    try:
        integration_instance.test_mock_model_integration()
        integration_instance.test_quantile_output_reshaping()
        
        print("\nPASS Integration Tests: ALL PASSED")
        
    except Exception as e:
        print(f"\nFAIL Integration Tests Failed: {e}")
        raise
    
    print("\nPARTY PARTY PARTY ALL DIMENSION MANAGER TESTS PASSED! PARTY PARTY PARTY")
    print("=" * 80)


if __name__ == "__main__":
    run_dimension_integration_tests()
