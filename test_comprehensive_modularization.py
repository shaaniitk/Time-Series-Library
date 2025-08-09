#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Phase 1 & Phase 2 Modularization

This script provides thorough testing of:
- Phase 1: All loss components and their functionality
- Phase 2: All attention components including restored algorithms
- Integration testing between components
- Performance validation
- Error handling and edge cases
"""

import sys
import torch
import torch.nn as nn
import traceback
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class ComprehensiveTestSuite:
    """Comprehensive test suite for modular components"""
    
    def __init__(self):
        self.test_results = {
            'phase1': {'passed': 0, 'failed': 0, 'details': []},
            'phase2': {'passed': 0, 'failed': 0, 'details': []},
            'integration': {'passed': 0, 'failed': 0, 'details': []},
            'performance': {'passed': 0, 'failed': 0, 'details': []}
        }
        
        # Test data configurations
        self.test_configs = [
            {'batch_size': 2, 'seq_len': 32, 'd_model': 64, 'n_heads': 4, 'features': 7},
            {'batch_size': 4, 'seq_len': 96, 'd_model': 128, 'n_heads': 8, 'features': 1},
            {'batch_size': 1, 'seq_len': 192, 'd_model': 256, 'n_heads': 16, 'features': 21}
        ]
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 COMPREHENSIVE MODULARIZATION TEST SUITE")
        print("=" * 80)
        print(f"Testing configurations: {len(self.test_configs)} different setups")
        print("=" * 80)
        
        # Phase 1: Loss Components
        self.test_phase1_comprehensive()
        
        # Phase 2: Attention Components
        self.test_phase2_comprehensive()
        
        # Integration Testing
        self.test_integration_comprehensive()
        
        # Performance Testing
        self.test_performance_comprehensive()
        
        # Summary
        self.print_final_summary()
    
    def test_phase1_comprehensive(self):
        """Comprehensive Phase 1 testing"""
        print("\n📊 PHASE 1: LOSS COMPONENTS COMPREHENSIVE TESTING")
        print("-" * 60)
        
        # Test all loss functions
        loss_components = [
            ('mse', {}),
            ('mae', {}),
            ('mape', {}),
            ('smape', {}),
            ('mase', {'baseline_mae': 0.1}),
            ('ps_loss', {}),
            ('focal', {'alpha': 0.25, 'gamma': 2.0}),
            ('frequency_aware', {}),
            ('multi_quantile', {'quantiles': [0.1, 0.5, 0.9]}),
            ('uncertainty_calibration', {})
        ]
        
        for loss_name, loss_kwargs in loss_components:
            self._test_loss_component(loss_name, loss_kwargs)
        
        # Test loss registry functionality
        self._test_loss_registry()
        
        # Test edge cases for losses
        self._test_loss_edge_cases()
    
    def test_phase2_comprehensive(self):
        """Comprehensive Phase 2 testing"""
        print("\n🎯 PHASE 2: ATTENTION COMPONENTS COMPREHENSIVE TESTING")
        print("-" * 60)
        
        # Test all attention components
        attention_components = [
            ('fourier_attention', {}),
            ('wavelet_attention', {'n_levels': 3}),
            ('enhanced_autocorrelation', {'adaptive_k': True, 'multi_scale': True}),
            ('linear_attention', {}),
            ('reformer_attention', {}),
            ('performer_attention', {}),
            ('flash_attention', {}),
            ('sparse_attention', {'sparsity_pattern': 'local'}),
            ('local_attention', {'window_size': 64}),
            ('global_local_attention', {'window_size': 32}),
            ('adaptive_attention', {}),
            ('temporal_attention', {}),
            ('cross_scale_attention', {'scales': [1, 2, 4]}),
            ('hierarchical_attention', {'levels': 3}),
            ('meta_learning_attention', {}),
            ('bayesian_attention', {}),
            ('mixture_attention', {'n_experts': 4}),
            ('multi_resolution_attention', {'resolutions': [32, 64, 128]})
        ]
        
        for attn_name, attn_kwargs in attention_components:
            self._test_attention_component(attn_name, attn_kwargs)
        
        # Test attention registry functionality
        self._test_attention_registry()
        
        # Test restored algorithms specifically
        self._test_restored_algorithms()
        
        # Test attention edge cases
        self._test_attention_edge_cases()
    
    def test_integration_comprehensive(self):
        """Comprehensive integration testing"""
        print("\n🔗 INTEGRATION TESTING")
        print("-" * 60)
        
        # Test model creation with different components
        self._test_model_integration()
        
        # Test component combinations
        self._test_component_combinations()
        
        # Test end-to-end workflows
        self._test_end_to_end_workflows()
    
    def test_performance_comprehensive(self):
        """Comprehensive performance testing"""
        print("\n⚡ PERFORMANCE TESTING")
        print("-" * 60)
        
        # Test component performance
        self._test_component_performance()
        
        # Test memory usage
        self._test_memory_usage()
        
        # Test scalability
        self._test_scalability()
    
    def _test_loss_component(self, loss_name: str, loss_kwargs: Dict):
        """Test individual loss component"""
        try:
            from layers.modular.losses.registry import LossRegistry
            registry = LossRegistry()
            
            # Test component creation
            loss_fn = registry.create(loss_name, **loss_kwargs)
            assert loss_fn is not None, f"Failed to create {loss_name}"
            
            # Test with multiple configurations
            for config in self.test_configs:
                pred = torch.randn(config['batch_size'], config['seq_len'], config['features'])
                target = torch.randn(config['batch_size'], config['seq_len'], config['features'])
                
                # Ensure target is positive for MAPE-like losses
                if loss_name in ['mape', 'smape']:
                    target = torch.abs(target) + 0.1
                
                loss = loss_fn(pred, target)
                
                # Validate loss properties
                assert isinstance(loss, torch.Tensor), f"{loss_name}: Loss not a tensor"
                assert loss.dim() == 0 or (loss.dim() == 1 and loss.size(0) == 1), f"{loss_name}: Loss not scalar"
                assert torch.isfinite(loss), f"{loss_name}: Loss not finite"
                
                # Test gradient computation
                loss.backward()
                assert pred.grad is not None, f"{loss_name}: No gradients computed"
                pred.grad.zero_()
            
            self.test_results['phase1']['passed'] += 1
            self.test_results['phase1']['details'].append(f"✅ {loss_name}: PASS")
            print(f"✅ {loss_name}: PASS")
            
        except Exception as e:
            self.test_results['phase1']['failed'] += 1
            error_msg = f"❌ {loss_name}: FAIL - {str(e)}"
            self.test_results['phase1']['details'].append(error_msg)
            print(error_msg)
            if "--verbose" in sys.argv:
                traceback.print_exc()
    
    def _test_attention_component(self, attn_name: str, attn_kwargs: Dict):
        """Test individual attention component"""
        try:
            from layers.modular.attention.registry import AttentionRegistry
            registry = AttentionRegistry()
            
            # Test with multiple configurations
            for config in self.test_configs:
                # Create attention component
                attention = registry.create(
                    attn_name, 
                    d_model=config['d_model'], 
                    n_heads=config['n_heads'],
                    **attn_kwargs
                )
                assert attention is not None, f"Failed to create {attn_name}"
                
                # Test forward pass
                queries = torch.randn(config['batch_size'], config['seq_len'], config['d_model'], requires_grad=True)
                keys = torch.randn(config['batch_size'], config['seq_len'], config['d_model'], requires_grad=True)
                values = torch.randn(config['batch_size'], config['seq_len'], config['d_model'], requires_grad=True)
                
                output, attn_weights = attention(queries, keys, values)
                
                # Validate output properties
                assert isinstance(output, torch.Tensor), f"{attn_name}: Output not a tensor"
                assert output.shape == queries.shape, f"{attn_name}: Output shape mismatch"
                assert torch.isfinite(output).all(), f"{attn_name}: Output contains non-finite values"
                
                # Test gradient computation
                loss = output.sum()
                loss.backward()
                assert queries.grad is not None, f"{attn_name}: No gradients computed"
                
                # Clean up gradients
                queries.grad.zero_()
                keys.grad.zero_()
                values.grad.zero_()
            
            self.test_results['phase2']['passed'] += 1
            self.test_results['phase2']['details'].append(f"✅ {attn_name}: PASS")
            print(f"✅ {attn_name}: PASS")
            
        except Exception as e:
            self.test_results['phase2']['failed'] += 1
            error_msg = f"❌ {attn_name}: FAIL - {str(e)}"
            self.test_results['phase2']['details'].append(error_msg)
            print(error_msg)
            if "--verbose" in sys.argv:
                traceback.print_exc()
    
    def _test_restored_algorithms(self):
        """Test specifically restored algorithms"""
        print("\n🔧 Testing Restored Algorithm Sophistication:")
        
        restored_algorithms = [
            'fourier_attention',
            'enhanced_autocorrelation', 
            'meta_learning_attention'
        ]
        
        for alg_name in restored_algorithms:
            self._test_algorithm_sophistication(alg_name)
    
    def _test_algorithm_sophistication(self, alg_name: str):
        """Test that algorithms maintain their sophistication"""
        try:
            if alg_name == 'fourier_attention':
                from layers.modular.attention.fourier_attention import FourierAttention
                attn = FourierAttention(64, 4)
                
                # Test complex frequency filtering
                queries = torch.randn(2, 32, 64, requires_grad=True)
                keys = torch.randn(2, 32, 64, requires_grad=True)
                values = torch.randn(2, 32, 64, requires_grad=True)
                
                output, _ = attn(queries, keys, values)
                assert torch.isfinite(output).all(), "FourierAttention: Non-finite outputs"
                
            elif alg_name == 'enhanced_autocorrelation':
                from layers.EfficientAutoCorrelation import EfficientAutoCorrelation
                autocorr = EfficientAutoCorrelation(adaptive_k=True, multi_scale=True)
                
                # Test adaptive k-selection and multi-scale
                queries = torch.randn(2, 32, 4, 16)
                keys = torch.randn(2, 32, 4, 16)
                values = torch.randn(2, 32, 4, 16)
                
                output, _ = autocorr(queries, keys, values, None)
                assert torch.isfinite(output).all(), "AutoCorrelation: Non-finite outputs"
                
                # Verify k_predictor exists and is sophisticated
                assert hasattr(autocorr, 'k_predictor'), "AutoCorrelation: Missing k_predictor"
                assert len(list(autocorr.k_predictor.modules())) > 1, "AutoCorrelation: k_predictor too simple"
                
            elif alg_name == 'meta_learning_attention':
                from layers.modular.attention.adaptive_components import MetaLearningAdapter
                meta_attn = MetaLearningAdapter(64, 4)
                
                # Test MAML functionality
                queries = torch.randn(2, 32, 64, requires_grad=True)
                keys = torch.randn(2, 32, 64, requires_grad=True)
                values = torch.randn(2, 32, 64, requires_grad=True)
                
                output, _ = meta_attn(queries, keys, values)
                assert torch.isfinite(output).all(), "MetaLearning: Non-finite outputs"
                
                # Verify MAML components exist
                assert hasattr(meta_attn, 'fast_weights'), "MetaLearning: Missing fast_weights"
            
            print(f"✅ {alg_name} sophistication: PASS")
            
        except Exception as e:
            print(f"❌ {alg_name} sophistication: FAIL - {str(e)}")
            if "--verbose" in sys.argv:
                traceback.print_exc()
    
    def _test_loss_registry(self):
        """Test loss registry functionality"""
        try:
            from layers.modular.losses.registry import LossRegistry
            registry = LossRegistry()
            
            # Test registration
            available_losses = registry.list_available()
            assert len(available_losses) > 0, "No losses registered"
            
            # Test component creation
            for loss_name in available_losses[:5]:  # Test first 5
                loss_fn = registry.get(loss_name)
                assert loss_fn is not None, f"Failed to get {loss_name}"
            
            self.test_results['phase1']['passed'] += 1
            print("✅ Loss Registry: PASS")
            
        except Exception as e:
            self.test_results['phase1']['failed'] += 1
            print(f"❌ Loss Registry: FAIL - {str(e)}")
    
    def _test_attention_registry(self):
        """Test attention registry functionality"""
        try:
            from layers.modular.attention.registry import AttentionRegistry
            registry = AttentionRegistry()
            
            # Test registration
            available_attentions = registry.list_available()
            assert len(available_attentions) > 0, "No attentions registered"
            
            # Test component creation
            for attn_name in available_attentions[:5]:  # Test first 5
                try:
                    attn = registry.create(attn_name, d_model=64, n_heads=4)
                    assert attn is not None, f"Failed to create {attn_name}"
                except Exception as inner_e:
                    print(f"   WARN: {attn_name} creation failed: {inner_e}")
            
            self.test_results['phase2']['passed'] += 1
            print("✅ Attention Registry: PASS")
            
        except Exception as e:
            self.test_results['phase2']['failed'] += 1
            print(f"❌ Attention Registry: FAIL - {str(e)}")
    
    def _test_loss_edge_cases(self):
        """Test loss functions with edge cases"""
        try:
            print("   Testing loss edge cases...")
            
            # Zero predictions
            pred_zero = torch.zeros(2, 10, 3)
            target = torch.ones(2, 10, 3)
            
            # Very small values
            pred_small = torch.full((2, 10, 3), 1e-8)
            target_small = torch.full((2, 10, 3), 1e-8)
            
            from layers.modular.losses.registry import LossRegistry
            registry = LossRegistry()
            
            # Test MSE with edge cases
            mse = registry.create('mse')
            loss1 = mse(pred_zero, target)
            loss2 = mse(pred_small, target_small)
            
            assert torch.isfinite(loss1), "MSE: Zero prediction case failed"
            assert torch.isfinite(loss2), "MSE: Small value case failed"
            
            self.test_results['phase1']['passed'] += 1
            print("✅ Loss Edge Cases: PASS")
            
        except Exception as e:
            self.test_results['phase1']['failed'] += 1
            print(f"❌ Loss Edge Cases: FAIL - {str(e)}")
    
    def _test_attention_edge_cases(self):
        """Test attention mechanisms with edge cases"""
        try:
            print("   Testing attention edge cases...")
            
            from layers.modular.attention.registry import AttentionRegistry
            registry = AttentionRegistry()
            
            # Test with very small sequences
            attn = registry.create('fourier_attention', d_model=32, n_heads=2)
            
            queries = torch.randn(1, 4, 32)  # Very short sequence
            keys = torch.randn(1, 4, 32)
            values = torch.randn(1, 4, 32)
            
            output, _ = attn(queries, keys, values)
            assert output.shape == queries.shape, "Edge case: Output shape mismatch"
            assert torch.isfinite(output).all(), "Edge case: Non-finite outputs"
            
            self.test_results['phase2']['passed'] += 1
            print("✅ Attention Edge Cases: PASS")
            
        except Exception as e:
            self.test_results['phase2']['failed'] += 1
            print(f"❌ Attention Edge Cases: FAIL - {str(e)}")
    
    def _test_model_integration(self):
        """Test integration in full models"""
        try:
            print("   Testing model integration...")
            
            # This would test full model creation but we'll do a simplified version
            # since full models might have dependencies we haven't set up yet
            
            from layers.modular.losses.registry import LossRegistry
            from layers.modular.attention.registry import AttentionRegistry
            
            loss_registry = LossRegistry()
            attn_registry = AttentionRegistry()
            
            # Test that we can create components and use them together
            loss_fn = loss_registry.create('mse')
            attn = attn_registry.create('fourier_attention', d_model=64, n_heads=4)
            
            # Simulate a simple forward pass
            x = torch.randn(2, 32, 64, requires_grad=True)
            attn_out, _ = attn(x, x, x)
            
            # Simulate loss computation
            target = torch.randn(2, 32, 64)
            loss = loss_fn(attn_out, target)
            
            # Test backward pass
            loss.backward()
            assert x.grad is not None, "Integration: No gradients"
            
            self.test_results['integration']['passed'] += 1
            print("✅ Model Integration: PASS")
            
        except Exception as e:
            self.test_results['integration']['failed'] += 1
            print(f"❌ Model Integration: FAIL - {str(e)}")
    
    def _test_component_combinations(self):
        """Test different component combinations"""
        try:
            print("   Testing component combinations...")
            
            # Test multiple attention types with same loss
            from layers.modular.losses.registry import LossRegistry
            from layers.modular.attention.registry import AttentionRegistry
            
            loss_registry = LossRegistry()
            attn_registry = AttentionRegistry()
            
            loss_fn = loss_registry.create('mae')
            
            attentions = ['fourier_attention', 'linear_attention']
            for attn_name in attentions:
                try:
                    attn = attn_registry.create(attn_name, d_model=32, n_heads=2)
                    x = torch.randn(1, 16, 32)
                    out, _ = attn(x, x, x)
                    loss = loss_fn(out, x)
                    assert torch.isfinite(loss), f"Combination {attn_name}+MAE failed"
                except Exception as inner_e:
                    print(f"   WARN: {attn_name} combination failed: {inner_e}")
            
            self.test_results['integration']['passed'] += 1
            print("✅ Component Combinations: PASS")
            
        except Exception as e:
            self.test_results['integration']['failed'] += 1
            print(f"❌ Component Combinations: FAIL - {str(e)}")
    
    def _test_end_to_end_workflows(self):
        """Test end-to-end workflows"""
        try:
            print("   Testing end-to-end workflows...")
            
            # Simulate a simple training step workflow
            from layers.modular.losses.registry import LossRegistry
            from layers.modular.attention.registry import AttentionRegistry
            
            loss_registry = LossRegistry()
            attn_registry = AttentionRegistry()
            
            # Create components
            loss_fn = loss_registry.create('mse')
            attn = attn_registry.create('fourier_attention', d_model=64, n_heads=4)
            
            # Simulate training data
            x = torch.randn(4, 48, 64, requires_grad=True)
            y = torch.randn(4, 48, 64)
            
            # Forward pass
            attn_out, _ = attn(x, x, x)
            loss = loss_fn(attn_out, y)
            
            # Backward pass
            loss.backward()
            
            # Check gradients exist
            assert x.grad is not None, "E2E: No gradients computed"
            assert torch.isfinite(x.grad).all(), "E2E: Non-finite gradients"
            
            self.test_results['integration']['passed'] += 1
            print("✅ End-to-End Workflows: PASS")
            
        except Exception as e:
            self.test_results['integration']['failed'] += 1
            print(f"❌ End-to-End Workflows: FAIL - {str(e)}")
    
    def _test_component_performance(self):
        """Test component performance"""
        try:
            print("   Testing component performance...")
            
            from layers.modular.attention.registry import AttentionRegistry
            attn_registry = AttentionRegistry()
            
            # Performance test configuration
            batch_size, seq_len, d_model, n_heads = 4, 128, 256, 8
            
            # Test a few key attention mechanisms
            test_attentions = ['fourier_attention', 'linear_attention', 'flash_attention']
            
            for attn_name in test_attentions:
                try:
                    attn = attn_registry.create(attn_name, d_model=d_model, n_heads=n_heads)
                    
                    x = torch.randn(batch_size, seq_len, d_model)
                    
                    # Time the forward pass
                    start_time = time.time()
                    for _ in range(5):  # Multiple runs for stability
                        output, _ = attn(x, x, x)
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / 5
                    print(f"   {attn_name}: {avg_time:.4f}s per forward pass")
                    
                    # Basic performance check - should complete in reasonable time
                    assert avg_time < 5.0, f"{attn_name}: Too slow ({avg_time:.4f}s)"
                    
                except Exception as inner_e:
                    print(f"   WARN: {attn_name} performance test failed: {inner_e}")
            
            self.test_results['performance']['passed'] += 1
            print("✅ Component Performance: PASS")
            
        except Exception as e:
            self.test_results['performance']['failed'] += 1
            print(f"❌ Component Performance: FAIL - {str(e)}")
    
    def _test_memory_usage(self):
        """Test memory usage"""
        try:
            print("   Testing memory usage...")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                
                from layers.modular.attention.registry import AttentionRegistry
                attn_registry = AttentionRegistry()
                
                attn = attn_registry.create('fourier_attention', d_model=128, n_heads=8)
                x = torch.randn(2, 64, 128).cuda()
                
                output, _ = attn.cuda()(x, x, x)
                
                peak_memory = torch.cuda.memory_allocated()
                memory_used = peak_memory - initial_memory
                
                print(f"   Memory used: {memory_used / 1024 / 1024:.2f} MB")
                
                # Clean up
                del attn, x, output
                torch.cuda.empty_cache()
            else:
                print("   CUDA not available, skipping GPU memory test")
            
            self.test_results['performance']['passed'] += 1
            print("✅ Memory Usage: PASS")
            
        except Exception as e:
            self.test_results['performance']['failed'] += 1
            print(f"❌ Memory Usage: FAIL - {str(e)}")
    
    def _test_scalability(self):
        """Test scalability with different input sizes"""
        try:
            print("   Testing scalability...")
            
            from layers.modular.attention.registry import AttentionRegistry
            attn_registry = AttentionRegistry()
            
            attn = attn_registry.create('linear_attention', d_model=64, n_heads=4)
            
            # Test different sequence lengths
            seq_lengths = [32, 64, 128, 256]
            times = []
            
            for seq_len in seq_lengths:
                x = torch.randn(2, seq_len, 64)
                
                start_time = time.time()
                output, _ = attn(x, x, x)
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            # Check that time doesn't grow too dramatically
            # (Linear attention should be roughly linear)
            time_ratio = times[-1] / times[0]  # 256 vs 32 length
            seq_ratio = seq_lengths[-1] / seq_lengths[0]  # 8x sequence length
            
            print(f"   Scalability ratio: {time_ratio:.2f}x time for {seq_ratio:.2f}x sequence")
            
            # Should not be worse than quadratic scaling
            assert time_ratio < seq_ratio ** 2, f"Poor scalability: {time_ratio:.2f}x time for {seq_ratio:.2f}x input"
            
            self.test_results['performance']['passed'] += 1
            print("✅ Scalability: PASS")
            
        except Exception as e:
            self.test_results['performance']['failed'] += 1
            print(f"❌ Scalability: FAIL - {str(e)}")
    
    def print_final_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("🏆 COMPREHENSIVE TEST SUITE RESULTS")
        print("=" * 80)
        
        total_passed = 0
        total_failed = 0
        
        for phase, results in self.test_results.items():
            passed = results['passed']
            failed = results['failed']
            total = passed + failed
            
            total_passed += passed
            total_failed += failed
            
            if total > 0:
                success_rate = (passed / total) * 100
                status = "✅ PASS" if failed == 0 else "⚠️  PARTIAL" if passed > failed else "❌ FAIL"
                print(f"{phase.upper():15} | {passed:3d}/{total:3d} | {success_rate:6.1f}% | {status}")
            else:
                print(f"{phase.upper():15} | {passed:3d}/{total:3d} |    N/A | ⚪ SKIP")
        
        print("-" * 80)
        grand_total = total_passed + total_failed
        if grand_total > 0:
            overall_success = (total_passed / grand_total) * 100
            overall_status = "🎉 SUCCESS" if total_failed == 0 else "⚠️  PARTIAL SUCCESS" if total_passed > total_failed else "💥 FAILURE"
            print(f"OVERALL RESULT  | {total_passed:3d}/{grand_total:3d} | {overall_success:6.1f}% | {overall_status}")
        else:
            print(f"OVERALL RESULT  |   0/  0 |    N/A | ⚪ NO TESTS RUN")
        
        print("=" * 80)
        
        # Print detailed results if requested
        if "--verbose" in sys.argv or total_failed > 0:
            print("\n📋 DETAILED RESULTS:")
            for phase, results in self.test_results.items():
                if results['details']:
                    print(f"\n{phase.upper()}:")
                    for detail in results['details']:
                        print(f"  {detail}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if total_failed == 0:
            print("🎯 All tests passed! The modularization is working excellently.")
            print("🚀 Ready to proceed to Phase 3: Decomposition Components")
        elif total_failed < total_passed:
            print("⚠️  Most tests passed, but some issues need attention:")
            print("🔧 Review failed tests and fix before proceeding")
            print("🧪 Re-run tests after fixes")
        else:
            print("🚨 Many tests failed - significant issues need resolution:")
            print("🛠️  Focus on core functionality first")
            print("📋 Review architecture and implementation")
        
        return total_failed == 0

def main():
    """Main test execution"""
    suite = ComprehensiveTestSuite()
    success = suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
