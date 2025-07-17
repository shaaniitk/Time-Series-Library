#!/usr/bin/env python3
"""
DimensionManager Utility Tests

Unit tests for the DimensionManager utility class, focusing on edge cases,
performance, and integration scenarios that commonly cause issues in
multi-time series applications.
"""

import pytest
import sys
import os
from typing import List, Dict
import dataclasses

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from utils.dimension_manager import DimensionManager


class TestDimensionManagerUtils:
    """Unit tests for DimensionManager utility functions"""
    
    def test_basic_instantiation(self):
        """Test basic DimensionManager instantiation"""
        dm = DimensionManager(
            mode='S',
            target_features=['price'],
            all_features=['price'],
            loss_function='mse'
        )
        
        assert dm.mode == 'S'
        assert dm.target_features == ['price']
        assert dm.all_features == ['price']
        assert dm.loss_function == 'mse'
        assert dm.quantiles == []
    
    def test_covariate_calculation(self):
        """Test covariate feature calculation"""
        targets = ['price', 'volume']
        all_features = ['price', 'volume', 'rsi', 'macd']
        
        dm = DimensionManager(
            mode='MS',
            target_features=targets,
            all_features=all_features,
            loss_function='mse'
        )
        
        expected_covariates = ['rsi', 'macd']
        assert dm.covariate_features == expected_covariates
        assert dm.num_targets == 2
        assert dm.num_covariates == 2
    
    def test_mode_s_dimensions(self):
        """Test dimension calculations for univariate mode (S)"""
        dm = DimensionManager(
            mode='S',
            target_features=['price'],
            all_features=['price', 'volume'],
            loss_function='mse'
        )
        
        # In S mode, only target features are used
        assert dm.enc_in == 1  # Only price
        assert dm.dec_in == 1  # Only price
        assert dm.c_out_evaluation == 1  # Only price
        assert dm.c_out_model == 1  # No quantile scaling
    
    def test_mode_ms_dimensions(self):
        """Test dimension calculations for multivariate-to-univariate mode (MS)"""
        targets = ['price']
        all_features = ['price', 'volume', 'rsi']
        
        dm = DimensionManager(
            mode='MS',
            target_features=targets,
            all_features=all_features,
            loss_function='mse'
        )
        
        # In MS mode, all features for input, only targets for output
        assert dm.enc_in == 3  # All features
        assert dm.dec_in == 3  # All features
        assert dm.c_out_evaluation == 1  # Only targets
        assert dm.c_out_model == 1  # No quantile scaling
    
    def test_mode_m_dimensions(self):
        """Test dimension calculations for multivariate mode (M)"""
        targets = ['price', 'volume']
        all_features = ['price', 'volume', 'rsi', 'macd']
        
        dm = DimensionManager(
            mode='M',
            target_features=targets,
            all_features=all_features,
            loss_function='mse'
        )
        
        # In M mode, all features for both input and output
        assert dm.enc_in == 4  # All features
        assert dm.dec_in == 4  # All features
        assert dm.c_out_evaluation == 4  # All features
        assert dm.c_out_model == 4  # No quantile scaling
    
    def test_quantile_scaling(self):
        """Test quantile loss dimension scaling"""
        quantiles = [0.1, 0.5, 0.9]
        
        dm = DimensionManager(
            mode='MS',
            target_features=['price'],
            all_features=['price', 'volume'],
            loss_function='quantile',
            quantiles=quantiles
        )
        
        # Base evaluation dimension should be number of targets
        assert dm.c_out_evaluation == 1
        
        # Model output should be scaled by number of quantiles
        assert dm.c_out_model == 1 * len(quantiles)
        assert dm.c_out_model == 3
    
    def test_pinball_loss_scaling(self):
        """Test pinball loss dimension scaling (same as quantile)"""
        quantiles = [0.25, 0.5, 0.75]
        
        dm = DimensionManager(
            mode='M',
            target_features=['price', 'volume'],
            all_features=['price', 'volume', 'rsi'],
            loss_function='pinball',
            quantiles=quantiles
        )
        
        # Base evaluation dimension
        assert dm.c_out_evaluation == 3  # All features in M mode
        
        # Model output scaled by quantiles
        assert dm.c_out_model == 3 * len(quantiles)
        assert dm.c_out_model == 9
    
    def test_empty_quantiles_with_quantile_loss(self):
        """Test behavior with quantile loss but empty quantiles list"""
        dm = DimensionManager(
            mode='S',
            target_features=['price'],
            all_features=['price'],
            loss_function='quantile',
            quantiles=[]  # Empty quantiles
        )
        
        # Should not scale dimensions if no quantiles provided
        assert dm.c_out_evaluation == 1
        assert dm.c_out_model == 1  # No scaling
    
    def test_non_quantile_loss_ignores_quantiles(self):
        """Test that non-quantile losses ignore quantiles parameter"""
        quantiles = [0.1, 0.5, 0.9]
        
        dm = DimensionManager(
            mode='MS',
            target_features=['price'],
            all_features=['price', 'volume'],
            loss_function='mse',  # Not a quantile loss
            quantiles=quantiles
        )
        
        # Should ignore quantiles for non-quantile losses
        assert dm.c_out_evaluation == 1
        assert dm.c_out_model == 1  # No scaling
    
    def test_repr_string(self):
        """Test string representation"""
        dm = DimensionManager(
            mode='MS',
            target_features=['price', 'volume'],
            all_features=['price', 'volume', 'rsi', 'macd'],
            loss_function='quantile',
            quantiles=[0.1, 0.5, 0.9]
        )
        
        repr_str = repr(dm)
        
        # Check that key information is included
        assert 'MS' in repr_str
        assert 'quantile' in repr_str
        assert '2 targets' in repr_str
        assert '2 covariates' in repr_str
        assert 'In=4' in repr_str
        assert 'Out(Layer)=6' in repr_str  # 2 targets * 3 quantiles
        assert 'Out(Eval)=2' in repr_str
    
    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError"""
        with pytest.raises(ValueError, match="Unknown feature mode"):
            DimensionManager(
                mode='INVALID',
                target_features=['price'],
                all_features=['price'],
                loss_function='mse'
            )
    
    def test_targets_not_in_all_features(self):
        """Test behavior when target features are not in all features"""
        # This should work in post_init when calculating covariate_features
        targets = ['missing_target']
        all_features = ['price', 'volume']
        
        dm = DimensionManager(
            mode='S',
            target_features=targets,
            all_features=all_features,
            loss_function='mse'
        )
        
        # The covariate calculation should handle this
        # missing_target is not in all_features, so all_features become covariates
        expected_covariates = ['price', 'volume']  # All features since target not found
        assert dm.covariate_features == expected_covariates
    
    def test_large_feature_sets(self):
        """Test with large numbers of features"""
        n_targets = 50
        n_total = 1000
        
        targets = [f'target_{i}' for i in range(n_targets)]
        all_features = targets + [f'cov_{i}' for i in range(n_total - n_targets)]
        
        dm = DimensionManager(
            mode='M',
            target_features=targets,
            all_features=all_features,
            loss_function='mse'
        )
        
        assert dm.num_targets == n_targets
        assert dm.num_covariates == n_total - n_targets
        assert dm.enc_in == n_total
        assert dm.c_out_evaluation == n_total
    
    def test_single_target_no_covariates(self):
        """Test minimal case: single target, no covariates"""
        dm = DimensionManager(
            mode='S',
            target_features=['price'],
            all_features=['price'],
            loss_function='mse'
        )
        
        assert dm.num_targets == 1
        assert dm.num_covariates == 0
        assert dm.enc_in == 1
        assert dm.dec_in == 1
        assert dm.c_out_evaluation == 1
        assert dm.c_out_model == 1
    
    def test_many_quantiles(self):
        """Test with a large number of quantiles"""
        quantiles = [i/100 for i in range(1, 100)]  # 99 quantiles
        
        dm = DimensionManager(
            mode='MS',
            target_features=['price'],
            all_features=['price', 'volume'],
            loss_function='quantile',
            quantiles=quantiles
        )
        
        assert dm.c_out_evaluation == 1
        assert dm.c_out_model == 99  # 1 target * 99 quantiles
    
    def test_dataclass_immutability(self):
        """Test that DimensionManager behaves as expected dataclass"""
        dm1 = DimensionManager(
            mode='S',
            target_features=['price'],
            all_features=['price'],
            loss_function='mse'
        )
        
        dm2 = DimensionManager(
            mode='S',
            target_features=['price'],
            all_features=['price'],
            loss_function='mse'
        )
        
        # Should be equal if same parameters
        assert dm1.mode == dm2.mode
        assert dm1.target_features == dm2.target_features
        assert dm1.all_features == dm2.all_features
        assert dm1.loss_function == dm2.loss_function
    
    def test_dimension_consistency_across_modes(self):
        """Test that dimension calculations are consistent across modes"""
        targets = ['price', 'volume']
        all_features = ['price', 'volume', 'rsi', 'macd', 'bollinger']
        
        configs = []
        for mode in ['S', 'MS', 'M']:
            dm = DimensionManager(
                mode=mode,
                target_features=targets,
                all_features=all_features,
                loss_function='mse'
            )
            configs.append(dm)
        
        # All should have same target/covariate counts
        for dm in configs:
            assert dm.num_targets == len(targets)
            assert dm.num_covariates == len(all_features) - len(targets)
        
        # Mode-specific dimension checks
        s_dm, ms_dm, m_dm = configs
        
        # S mode: only targets
        assert s_dm.enc_in == len(targets)
        assert s_dm.c_out_evaluation == len(targets)
        
        # MS mode: all features input, targets output
        assert ms_dm.enc_in == len(all_features)
        assert ms_dm.c_out_evaluation == len(targets)
        
        # M mode: all features input and output
        assert m_dm.enc_in == len(all_features)
        assert m_dm.c_out_evaluation == len(all_features)
    
    def test_quantile_mode_interactions(self):
        """Test interactions between quantile losses and different modes"""
        targets = ['price']
        all_features = ['price', 'volume', 'rsi']
        quantiles = [0.1, 0.5, 0.9]
        
        for mode in ['S', 'MS', 'M']:
            dm = DimensionManager(
                mode=mode,
                target_features=targets,
                all_features=all_features,
                loss_function='quantile',
                quantiles=quantiles
            )
            
            # Base evaluation dimensions should follow mode rules
            if mode == 'S':
                expected_eval = len(targets)
            elif mode == 'MS':
                expected_eval = len(targets)
            elif mode == 'M':
                expected_eval = len(all_features)
            
            assert dm.c_out_evaluation == expected_eval
            assert dm.c_out_model == expected_eval * len(quantiles)


def run_dimension_manager_utils_tests():
    """Run all DimensionManager utility tests"""
    
    print("🧪 Running DimensionManager Utility Tests")
    print("=" * 60)
    
    test_instance = TestDimensionManagerUtils()
    
    # List of test methods
    test_methods = [
        'test_basic_instantiation',
        'test_covariate_calculation',
        'test_mode_s_dimensions',
        'test_mode_ms_dimensions',
        'test_mode_m_dimensions',
        'test_quantile_scaling',
        'test_pinball_loss_scaling',
        'test_empty_quantiles_with_quantile_loss',
        'test_non_quantile_loss_ignores_quantiles',
        'test_repr_string',
        'test_invalid_mode_raises_error',
        'test_targets_not_in_all_features',
        'test_large_feature_sets',
        'test_single_target_no_covariates',
        'test_many_quantiles',
        'test_dataclass_immutability',
        'test_dimension_consistency_across_modes',
        'test_quantile_mode_interactions'
    ]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            print(f"  🧪 Running {method_name}...")
            method = getattr(test_instance, method_name)
            method()
            print(f"    ✅ {method_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"    ❌ {method_name} FAILED: {e}")
            failed += 1
    
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"📊 DimensionManager Utils Test Results:")
    print(f"   Total Tests: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if failed == 0:
        print("🎉 All DimensionManager utility tests passed!")
    else:
        print("⚠️ Some tests failed. Check individual test output for details.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_dimension_manager_utils_tests()
    sys.exit(0 if success else 1)
