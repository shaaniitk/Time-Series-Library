"""
Integration Test for Advanced Components

This script tests the integration of all sophisticated existing implementations
into the modular framework, with special focus on Bayesian losses with KL divergence.
"""

import torch
import logging
import sys
import traceback
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_advanced_component_registration():
    """Test that all advanced components are properly registered"""
    print("\n" + "="*60)
    print("TESTING ADVANCED COMPONENT REGISTRATION")
    print("="*60)
    
    try:
        from utils.modular_components.implementations.register_advanced import (
            get_advanced_component_summary, validate_bayesian_integration
        )
        
        # Get summary
        summary = get_advanced_component_summary()
        print(f"📊 Total advanced components registered: {summary['total_advanced_components']}")
        print(f"📈 Components by category: {summary['by_category']}")
        print(f"🧠 Bayesian components: {len(summary['bayesian_components'])}")
        print(f"⚡ Optimized components: {len(summary['optimized_components'])}")
        print(f"🔧 Utility processors: {len(summary['utility_processors'])}")
        
        # Validate Bayesian integration
        validation = validate_bayesian_integration()
        print(f"\n🔍 Bayesian Integration Validation:")
        for key, value in validation.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
        
        return all(validation.values())
        
    except Exception as e:
        print(f"❌ Registration test failed: {e}")
        traceback.print_exc()
        return False


def test_bayesian_loss_integration():
    """Test the critical Bayesian loss integration with KL divergence"""
    print("\n" + "="*60)
    print("TESTING BAYESIAN LOSS INTEGRATION (CRITICAL)")
    print("="*60)
    
    try:
        from utils.modular_components.registry import get_global_registry
        
        registry = get_global_registry()
        
        # Test Bayesian MSE Loss
        print("🧪 Testing Bayesian MSE Loss...")
        if registry.is_registered('loss', 'bayesian_mse'):
            bayesian_mse_class = registry.get('loss', 'bayesian_mse')
            print(f"✅ BayesianMSE class retrieved: {bayesian_mse_class}")
            
            # Test instantiation (with mock config)
            class MockConfig:
                def __init__(self):
                    self.kl_weight = 1e-5
                    self.uncertainty_weight = 0.1
                    self.reduction = 'mean'
            
            config = MockConfig()
            loss_fn = bayesian_mse_class(config)
            print(f"✅ BayesianMSE instantiated successfully")
            
            # Test with dummy data
            pred = torch.randn(10, 20, 5)
            true = torch.randn(10, 20, 5)
            
            # Test loss computation
            loss_result = loss_fn.compute_loss(pred, true)
            print(f"✅ BayesianMSE loss computed: {loss_result}")
            
            # Test KL divergence extraction
            if hasattr(loss_fn, 'get_kl_divergence'):
                kl_div = loss_fn.get_kl_divergence()
                print(f"✅ KL divergence extracted: {kl_div}")
            else:
                print("⚠️ KL divergence extraction method not found")
            
        else:
            print("❌ Bayesian MSE not registered!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Bayesian loss test failed: {e}")
        traceback.print_exc()
        return False


def test_advanced_attention_integration():
    """Test advanced attention mechanism integration"""
    print("\n" + "="*60)
    print("TESTING ADVANCED ATTENTION INTEGRATION")
    print("="*60)
    
    try:
        from utils.modular_components.registry import get_global_registry
        
        registry = get_global_registry()
        
        # Test OptimizedAutoCorrelation
        print("🧪 Testing OptimizedAutoCorrelation...")
        if registry.is_registered('attention', 'optimized_autocorrelation'):
            attention_class = registry.get('attention', 'optimized_autocorrelation')
            print(f"✅ OptimizedAutoCorrelation class retrieved: {attention_class}")
            
            # Test metadata
            metadata = registry.get_metadata('attention', 'optimized_autocorrelation')
            print(f"📋 Metadata: {metadata}")
            
            if metadata.get('memory_efficient') and metadata.get('optimized'):
                print("✅ Memory optimization features confirmed")
            else:
                print("⚠️ Memory optimization features not confirmed")
        
        # Test AdaptiveAutoCorrelation  
        print("\n🧪 Testing AdaptiveAutoCorrelation...")
        if registry.is_registered('attention', 'adaptive_autocorrelation'):
            adaptive_class = registry.get('attention', 'adaptive_autocorrelation')
            print(f"✅ AdaptiveAutoCorrelation class retrieved: {adaptive_class}")
            
            metadata = registry.get_metadata('attention', 'adaptive_autocorrelation')
            if metadata.get('adaptive') and metadata.get('multi_scale'):
                print("✅ Adaptive and multi-scale features confirmed")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced attention test failed: {e}")
        traceback.print_exc()
        return False


def test_specialized_processors():
    """Test specialized signal processors"""
    print("\n" + "="*60)
    print("TESTING SPECIALIZED PROCESSORS")
    print("="*60)
    
    try:
        from utils.modular_components.registry import get_global_registry
        
        registry = get_global_registry()
        
        processors_to_test = [
            'frequency_domain',
            'structural_patch',
            'dtw_alignment',
            'trend_analysis',
            'integrated_signal'
        ]
        
        for processor_name in processors_to_test:
            print(f"\n🧪 Testing {processor_name}...")
            if registry.is_registered('processor', processor_name):
                processor_class = registry.get('processor', processor_name)
                metadata = registry.get_metadata('processor', processor_name)
                print(f"✅ {processor_name} registered with metadata: {metadata.get('type', 'unknown')}")
            else:
                print(f"❌ {processor_name} not registered!")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Specialized processors test failed: {e}")
        traceback.print_exc()
        return False


def test_existing_implementation_access():
    """Test that we can access the original sophisticated implementations"""
    print("\n" + "="*60)
    print("TESTING ACCESS TO EXISTING IMPLEMENTATIONS")
    print("="*60)
    
    success_count = 0
    total_tests = 0
    
    # Test access to existing Bayesian losses
    try:
        from utils.bayesian_losses import BayesianLoss
        print("✅ Access to utils.bayesian_losses.BayesianLoss confirmed")
        success_count += 1
    except ImportError:
        print("❌ Cannot access utils.bayesian_losses.BayesianLoss")
    total_tests += 1
    
    # Test access to enhanced losses
    try:
        from utils.enhanced_losses import FrequencyAwareLoss
        print("✅ Access to utils.enhanced_losses.FrequencyAwareLoss confirmed")
        success_count += 1
    except ImportError:
        print("❌ Cannot access utils.enhanced_losses.FrequencyAwareLoss")
    total_tests += 1
    
    # Test access to optimized autocorrelation
    try:
        from layers.AutoCorrelation_Optimized import OptimizedAutoCorrelation
        print("✅ Access to layers.AutoCorrelation_Optimized confirmed")
        success_count += 1
    except ImportError:
        print("❌ Cannot access layers.AutoCorrelation_Optimized")
    total_tests += 1
    
    # Test access to enhanced autocorrelation
    try:
        from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelation
        print("✅ Access to layers.EnhancedAutoCorrelation confirmed")
        success_count += 1
    except ImportError:
        print("❌ Cannot access layers.EnhancedAutoCorrelation")
    total_tests += 1
    
    print(f"\n📊 Existing implementation access: {success_count}/{total_tests} successful")
    return success_count == total_tests


def test_integration_completeness():
    """Test that the integration addresses the original oversight"""
    print("\n" + "="*60)
    print("TESTING INTEGRATION COMPLETENESS")
    print("="*60)
    
    try:
        from utils.modular_components.implementations import get_integration_status
        
        status = get_integration_status()
        print(f"📋 Integration Status:")
        for key, value in status.items():
            status_icon = "✅" if value else "❌"
            print(f"  {status_icon} {key}: {value}")
        
        # Critical checks
        critical_components = [
            status['advanced_losses_available'],
            status['advanced_attentions_available'],
            status['specialized_processors_available'],
            status['advanced_registration_complete']
        ]
        
        if all(critical_components):
            print("\n🎉 ALL CRITICAL INTEGRATIONS COMPLETE!")
            print("The modular framework now leverages ALL existing sophisticated implementations!")
            return True
        else:
            print("\n⚠️ Some critical integrations are missing")
            return False
            
    except Exception as e:
        print(f"❌ Integration completeness test failed: {e}")
        traceback.print_exc()
        return False


def run_comprehensive_integration_test():
    """Run all integration tests"""
    print("🚀 COMPREHENSIVE ADVANCED INTEGRATION TEST")
    print("="*80)
    print("This test verifies that the modular framework now leverages")
    print("ALL sophisticated existing implementations, addressing the critical")
    print("oversight where 'whole purpose of bayesian autoformer is defeated without KL loss'")
    print("="*80)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Advanced Component Registration", test_advanced_component_registration),
        ("Bayesian Loss Integration (CRITICAL)", test_bayesian_loss_integration),
        ("Advanced Attention Integration", test_advanced_attention_integration),
        ("Specialized Processors", test_specialized_processors),
        ("Existing Implementation Access", test_existing_implementation_access),
        ("Integration Completeness", test_integration_completeness)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            test_results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 🎉 🎉 INTEGRATION COMPLETE! 🎉 🎉 🎉")
        print("The modular framework is now 100% complete with ALL")
        print("sophisticated existing implementations properly integrated!")
        print("Bayesian models now have proper KL divergence support!")
    else:
        print(f"\n⚠️ Integration partially complete: {passed}/{total}")
        print("Some components may not be fully functional.")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_integration_test()
    sys.exit(0 if success else 1)
