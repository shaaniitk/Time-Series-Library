#!/usr/bin/env python3
"""
Model Comparison Unit Test

This script runs identical unit tests for both TimesNet and Autoformer
to provide a fair comparison of their performance on the same synthetic data.

Both models use exactly the same configuration:
- seq_len = 24, pred_len = 12, d_model = 16
- Same synthetic sine wave data
- Same training procedure and epochs
- Same evaluation metrics

This gives us a true apples-to-apples comparison.
"""

import unittest
import sys
import os
import torch
import time
from utils.logger import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelComparison:
    """Compare TimesNet and Autoformer on identical unit tests"""
    
    def __init__(self):
        self.results = {}
        
    def run_comparison(self):
        """Run comparison tests for both models"""
        logger.info("="*70)
        logger.info("MODEL COMPARISON: TIMESNET vs AUTOFORMER")
        logger.info("="*70)
        logger.info("Configuration: seq_len=24, pred_len=12, d_model=16")
        logger.info("Data: Synthetic sine waves with 3 targets + 3 covariates")
        logger.info("Training: 10 epochs with early stopping")
        logger.info("="*70)
        
        # Test TimesNet
        logger.info("\n🔵 TESTING TIMESNET")
        logger.info("-" * 40)
        timesnet_results = self._test_model('TimesNet')
        
        # Test Autoformer  
        logger.info("\n🟢 TESTING AUTOFORMER")
        logger.info("-" * 40)
        autoformer_results = self._test_model('Autoformer')
        
        # Compare results
        self._compare_results(timesnet_results, autoformer_results)
        
    def _test_model(self, model_name):
        """Test a specific model"""
        try:
            if model_name == 'TimesNet':
                from tests.models.test_timesnet import TestTimesNet
                test_class = TestTimesNet
            elif model_name == 'Autoformer':
                from tests.models.test_autoformer import TestAutoformer 
                test_class = TestAutoformer
            else:
                raise ValueError(f"Unknown model: {model_name}")
                
            # Create test instance
            test_instance = test_class()
            test_instance.setUp()
            
            # Test forward pass
            logger.info(f"⚡ Testing {model_name} forward pass...")
            start_time = time.time()
            test_instance.test_forward_shape()
            forward_time = time.time() - start_time
            logger.info(f"✓ Forward pass completed in {forward_time:.3f}s")
            
            # Test parameter count
            logger.info(f"📊 Counting {model_name} parameters...")
            test_instance.test_parameter_count()
            total_params = sum(p.numel() for p in test_instance.model.parameters())
            
            # Test sanity (main performance test)
            logger.info(f"🧪 Running {model_name} sanity test...")
            start_time = time.time()
            test_instance.test_sanity()
            sanity_time = time.time() - start_time
            
            # Extract MSE from the test (we need to run it again to get the value)
            mse, _, _ = test_instance.run_sanity_test(test_instance.model.__class__)
            
            results = {
                'model_name': model_name,
                'forward_time': forward_time,
                'sanity_time': sanity_time,
                'total_params': total_params,
                'final_mse': mse,
                'success': True
            }
            
            logger.info(f"✅ {model_name} testing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ {model_name} testing failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'model_name': model_name,
                'success': False,
                'error': str(e)
            }
    
    def _compare_results(self, timesnet_results, autoformer_results):
        """Compare and display results"""
        logger.info("\n" + "="*70)
        logger.info("📊 COMPARISON RESULTS")
        logger.info("="*70)
        
        if not timesnet_results['success'] or not autoformer_results['success']:
            logger.error("⚠️  Cannot compare - one or both tests failed")
            if not timesnet_results['success']:
                logger.error(f"TimesNet error: {timesnet_results.get('error', 'Unknown')}")
            if not autoformer_results['success']:
                logger.error(f"Autoformer error: {autoformer_results.get('error', 'Unknown')}")
            return
            
        # Extract results
        tn_params = timesnet_results['total_params']
        af_params = autoformer_results['total_params']
        tn_mse = timesnet_results['final_mse']
        af_mse = autoformer_results['final_mse']
        tn_time = timesnet_results['sanity_time']
        af_time = autoformer_results['sanity_time']
        
        # Display comparison table
        logger.info(f"{'Metric':<20} {'TimesNet':<15} {'Autoformer':<15} {'Winner':<10}")
        logger.info("-" * 70)
        logger.info(f"{'Parameters':<20} {tn_params:<15,} {af_params:<15,} {'TN' if tn_params < af_params else 'AF' if af_params < tn_params else 'Tie':<10}")
        logger.info(f"{'Final MSE':<20} {tn_mse:<15.6f} {af_mse:<15.6f} {'TN' if tn_mse < af_mse else 'AF' if af_mse < tn_mse else 'Tie':<10}")
        logger.info(f"{'Training Time (s)':<20} {tn_time:<15.3f} {af_time:<15.3f} {'TN' if tn_time < af_time else 'AF' if af_time < tn_time else 'Tie':<10}")
        
        # Calculate relative differences
        param_diff = abs(tn_params - af_params) / max(tn_params, af_params) * 100
        mse_diff = abs(tn_mse - af_mse) / max(tn_mse, af_mse) * 100
        time_diff = abs(tn_time - af_time) / max(tn_time, af_time) * 100
        
        logger.info("\n📈 RELATIVE DIFFERENCES:")
        logger.info(f"Parameters: {param_diff:.1f}% difference")
        logger.info(f"MSE: {mse_diff:.1f}% difference")
        logger.info(f"Training Time: {time_diff:.1f}% difference")
        
        # Determine overall winner
        tn_wins = 0
        af_wins = 0
        
        if tn_params < af_params: tn_wins += 1
        elif af_params < tn_params: af_wins += 1
        
        if tn_mse < af_mse: tn_wins += 1
        elif af_mse < tn_mse: af_wins += 1
        
        if tn_time < af_time: tn_wins += 1
        elif af_time < tn_time: af_wins += 1
        
        logger.info("\n🏆 OVERALL ASSESSMENT:")
        if tn_wins > af_wins:
            logger.info("🔵 TimesNet performs better overall")
        elif af_wins > tn_wins:
            logger.info("🟢 Autoformer performs better overall")
        else:
            logger.info("⚖️  Both models perform similarly")
            
        logger.info("="*70)


def main():
    """Main comparison function"""
    try:
        comparison = ModelComparison()
        comparison.run_comparison()
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
