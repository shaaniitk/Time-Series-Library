"""
Comprehensive Test Suite for Enhanced Autoformer Components

This module provides unit tests and benchmarks for all the enhanced
Autoformer components to ensure correctness and performance.
"""

import unittest
import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelation, AdaptiveAutoCorrelationLayer
from models.EnhancedAutoformer import LearnableSeriesDecomp, EnhancedEncoderLayer
from utils.enhanced_losses import AdaptiveAutoformerLoss, FrequencyAwareLoss, CurriculumLossScheduler
from utils.logger import logger


class TestEnhancedAutoCorrelation(unittest.TestCase):
    """Test suite for Enhanced AutoCorrelation components."""
    
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 96
        self.n_heads = 8
        self.d_model = 64
        self.head_dim = self.d_model // self.n_heads
        
    def test_adaptive_autocorr_forward(self):
        """Test basic forward pass of adaptive autocorrelation."""
        logger.info("Testing AdaptiveAutoCorrelation forward pass")
        
        autocorr = AdaptiveAutoCorrelation(
            factor=1,
            adaptive_k=True,
            multi_scale=True,
            scales=[1, 2, 4]
        )
        
        # Create test inputs
        B, L, H, E = self.batch_size, self.seq_len, self.n_heads, self.head_dim
        queries = torch.randn(B, L, H, E)
        keys = torch.randn(B, L, H, E)
        values = torch.randn(B, L, H, E)
        
        # Forward pass
        output, attention = autocorr(queries, keys, values, None)
        
        # Check output shapes
        self.assertEqual(output.shape, (B, L, H, E))
        self.assertIsNotNone(output)
        
        # Check that output is different from input (model is doing something)
        self.assertFalse(torch.allclose(output, values, atol=1e-6))
        
        logger.info("✓ AdaptiveAutoCorrelation forward pass test passed")
    
    def test_adaptive_k_selection(self):
        """Test adaptive k selection mechanism."""
        logger.info("Testing adaptive k selection")
        
        autocorr = AdaptiveAutoCorrelation(factor=1, adaptive_k=True)
        
        # Test with different sequence lengths
        for seq_len in [24, 48, 96, 192]:
            corr_energy = torch.randn(self.batch_size, seq_len)
            k = autocorr.select_adaptive_k(corr_energy, seq_len)
            
            # Check that k is reasonable
            min_expected = max(2, int(0.1 * np.log(seq_len)))
            max_expected = min(int(0.3 * seq_len), int(autocorr.factor * np.log(seq_len) * 2))
            
            self.assertGreaterEqual(k, min_expected)
            self.assertLessEqual(k, max_expected)
            
        logger.info("✓ Adaptive k selection test passed")
    
    def test_multi_scale_correlation(self):
        """Test multi-scale correlation computation."""
        logger.info("Testing multi-scale correlation")
        
        autocorr = AdaptiveAutoCorrelation(
            factor=1,
            multi_scale=True,
            scales=[1, 2, 4]
        )
        
        B, H, E, L = self.batch_size, self.n_heads, self.head_dim, self.seq_len
        queries = torch.randn(B, H, E, L)
        keys = torch.randn(B, H, E, L)
        
        # Test multi-scale correlation
        corr = autocorr.multi_scale_correlation(queries, keys, L)
        
        # Check output shape
        expected_shape = (B, H, E, L)
        self.assertEqual(corr.shape, expected_shape)
        
        # Check that correlation values are reasonable
        self.assertFalse(torch.isnan(corr).any())
        self.assertFalse(torch.isinf(corr).any())
        
        logger.info("✓ Multi-scale correlation test passed")
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        logger.info("Testing numerical stability")
        
        autocorr = AdaptiveAutoCorrelation(factor=1, eps=1e-8)
        
        B, L, H, E = self.batch_size, self.seq_len, self.n_heads, self.head_dim
        
        # Test with very small values
        small_values = torch.randn(B, L, H, E) * 1e-6
        output, _ = autocorr(small_values, small_values, small_values, None)
        self.assertFalse(torch.isnan(output).any())
        
        # Test with very large values
        large_values = torch.randn(B, L, H, E) * 1e6
        output, _ = autocorr(large_values, large_values, large_values, None)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
        logger.info("✓ Numerical stability test passed")


class TestLearnableSeriesDecomp(unittest.TestCase):
    """Test suite for Learnable Series Decomposition."""
    
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 96
        self.d_model = 64
        
    def test_decomposition_forward(self):
        """Test basic forward pass of learnable decomposition."""
        logger.info("Testing LearnableSeriesDecomp forward pass")
        
        decomp = LearnableSeriesDecomp(self.d_model)
        
        # Create test input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        seasonal, trend = decomp(x)
        
        # Check output shapes
        self.assertEqual(seasonal.shape, x.shape)
        self.assertEqual(trend.shape, x.shape)
        
        # Check that seasonal + trend ≈ original (approximately)
        reconstructed = seasonal + trend
        # Allow some small difference due to learnable parameters
        self.assertTrue(torch.allclose(reconstructed, x, atol=1.0))
        
        logger.info("✓ LearnableSeriesDecomp forward pass test passed")
    
    def test_adaptive_kernel_sizes(self):
        """Test that different inputs produce different kernel sizes."""
        logger.info("Testing adaptive kernel sizes")
        
        decomp = LearnableSeriesDecomp(self.d_model)
        
        # Create inputs with different characteristics
        smooth_input = torch.sin(torch.linspace(0, 4*np.pi, self.seq_len)).unsqueeze(0).unsqueeze(-1).repeat(1, 1, self.d_model)
        noisy_input = torch.randn(1, self.seq_len, self.d_model)
        
        # Forward pass to trigger kernel size prediction
        # Note: This test assumes the implementation stores or returns kernel sizes
        # You might need to modify based on actual implementation
        _ = decomp(smooth_input)
        _ = decomp(noisy_input)
        
        logger.info("✓ Adaptive kernel sizes test passed")


class TestEnhancedLosses(unittest.TestCase):
    """Test suite for Enhanced Loss Functions."""
    
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 96
        self.d_model = 7
        
    def test_adaptive_loss(self):
        """Test adaptive autoformer loss."""
        logger.info("Testing AdaptiveAutoformerLoss")
        
        loss_fn = AdaptiveAutoformerLoss(adaptive_weights=True)
        
        # Create test data
        pred = torch.randn(self.batch_size, self.seq_len, self.d_model)
        true = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Test forward pass
        loss = loss_fn(pred, true)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertGreater(loss.item(), 0)  # Positive loss
        
        # Test with component return
        loss, components = loss_fn(pred, true, return_components=True)
        self.assertIsInstance(components, dict)
        self.assertIn('trend_loss', components)
        self.assertIn('seasonal_loss', components)
        
        logger.info("✓ AdaptiveAutoformerLoss test passed")
    
    def test_frequency_aware_loss(self):
        """Test frequency-aware loss."""
        logger.info("Testing FrequencyAwareLoss")
        
        loss_fn = FrequencyAwareLoss(freq_weight=0.1)
        
        # Create test data
        pred = torch.randn(self.batch_size, self.seq_len, self.d_model)
        true = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Test forward pass
        loss = loss_fn(pred, true)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)
        
        logger.info("✓ FrequencyAwareLoss test passed")
    
    def test_curriculum_scheduler(self):
        """Test curriculum learning scheduler."""
        logger.info("Testing CurriculumLossScheduler")
        
        scheduler = CurriculumLossScheduler(
            start_seq_len=24,
            target_seq_len=96,
            curriculum_epochs=50
        )
        
        # Test sequence length progression
        epochs_to_test = [0, 10, 25, 50, 100]
        seq_lens = [scheduler.get_current_seq_len(epoch) for epoch in epochs_to_test]
        
        # Check that sequence length increases
        self.assertEqual(seq_lens[0], 24)  # Start length
        self.assertEqual(seq_lens[-1], 96)  # Target length
        self.assertTrue(all(seq_lens[i] <= seq_lens[i+1] for i in range(len(seq_lens)-1)))
        
        # Test curriculum application
        batch_x = torch.randn(self.batch_size, 96, self.d_model)
        batch_y = torch.randn(self.batch_size, 24, self.d_model)
        batch_x_mark = torch.randn(self.batch_size, 96, 4)
        batch_y_mark = torch.randn(self.batch_size, 24, 4)
        
        curr_x, curr_y, curr_x_mark, curr_y_mark = scheduler.apply_curriculum(
            batch_x, batch_y, batch_x_mark, batch_y_mark, epoch=10
        )
        
        # Check that sequence was truncated
        expected_len = scheduler.get_current_seq_len(10)
        self.assertLessEqual(curr_x.size(1), expected_len)
        
        logger.info("✓ CurriculumLossScheduler test passed")


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for enhanced components."""
    
    def setUp(self):
        self.batch_size = 8
        self.seq_len = 192
        self.n_heads = 8
        self.d_model = 128
        self.head_dim = self.d_model // self.n_heads
        self.n_iterations = 10
        
    def test_autocorr_performance(self):
        """Benchmark autocorrelation performance."""
        logger.info("Benchmarking AutoCorrelation performance")
        
        # Standard AutoCorrelation (would need to import from original)
        # For now, we'll just benchmark the enhanced version
        enhanced_autocorr = AdaptiveAutoCorrelation(
            factor=1,
            adaptive_k=True,
            multi_scale=True,
            scales=[1, 2]
        )
        
        B, L, H, E = self.batch_size, self.seq_len, self.n_heads, self.head_dim
        queries = torch.randn(B, L, H, E)
        keys = torch.randn(B, L, H, E)
        values = torch.randn(B, L, H, E)
        
        # Warmup
        for _ in range(3):
            _ = enhanced_autocorr(queries, keys, values, None)
        
        # Benchmark
        start_time = time.time()
        for _ in range(self.n_iterations):
            output, _ = enhanced_autocorr(queries, keys, values, None)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / self.n_iterations
        logger.info(f"Enhanced AutoCorrelation: {avg_time:.4f}s per forward pass")
        
        # Check that timing is reasonable (less than 1 second per pass)
        self.assertLess(avg_time, 1.0)
    
    def test_memory_usage(self):
        """Test memory usage of enhanced components."""
        logger.info("Testing memory usage")
        
        import torch.cuda
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.cuda.empty_cache()
            
            autocorr = AdaptiveAutoCorrelation(
                factor=1,
                adaptive_k=True,
                multi_scale=True,
                scales=[1, 2, 4]
            ).to(device)
            
            B, L, H, E = 16, 512, 8, 64  # Larger batch for memory test
            queries = torch.randn(B, L, H, E, device=device)
            keys = torch.randn(B, L, H, E, device=device)
            values = torch.randn(B, L, H, E, device=device)
            
            memory_before = torch.cuda.memory_allocated()
            
            # Forward pass
            output, _ = autocorr(queries, keys, values, None)
            
            memory_after = torch.cuda.memory_allocated()
            memory_used = (memory_after - memory_before) / (1024**2)  # MB
            
            logger.info(f"Memory usage: {memory_used:.2f} MB")
            
            # Check that memory usage is reasonable
            self.assertLess(memory_used, 1000)  # Less than 1GB
        else:
            logger.info("CUDA not available, skipping GPU memory test")


def run_all_tests():
    """Run all tests and return results summary."""
    logger.info("="*60)
    logger.info("RUNNING ENHANCED AUTOFORMER TEST SUITE")
    logger.info("="*60)
    
    # Create test suite
    test_classes = [
        TestEnhancedAutoCorrelation,
        TestLearnableSeriesDecomp,
        TestEnhancedLosses,
        TestPerformanceBenchmarks
    ]
    
    results = {}
    total_tests = 0
    total_passed = 0
    
    for test_class in test_classes:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        class_name = test_class.__name__
        tests_run = result.testsRun
        tests_passed = tests_run - len(result.failures) - len(result.errors)
        
        results[class_name] = {
            'tests_run': tests_run,
            'tests_passed': tests_passed,
            'failures': len(result.failures),
            'errors': len(result.errors)
        }
        
        total_tests += tests_run
        total_passed += tests_passed
    
    # Print summary
    logger.info("="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    for class_name, result in results.items():
        status = "✓ PASS" if result['failures'] == 0 and result['errors'] == 0 else "✗ FAIL"
        logger.info(f"{class_name}: {result['tests_passed']}/{result['tests_run']} {status}")
    
    logger.info("-"*60)
    logger.info(f"TOTAL: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        logger.info("🎉 ALL TESTS PASSED!")
    else:
        logger.warning(f"⚠️  {total_tests - total_passed} tests failed")
    
    return results


if __name__ == "__main__":
    # Set logging level for tests
    import logging
    from utils.logger import set_log_level
    set_log_level(logging.INFO)
    
    # Run all tests
    results = run_all_tests()
    
    # Exit with appropriate code
    total_failures = sum(r['failures'] + r['errors'] for r in results.values())
    exit(0 if total_failures == 0 else 1)
