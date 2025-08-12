#!/usr/bin/env python3
"""
Utils System Assessment & Compatibility Testing

Phase 1 of registry consolidation: Test utils/ system capabilities
and compatibility with our approach.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_utils_system_capabilities():
    """Test the utils/ modular system capabilities"""
    print("🔍 PHASE 1: UTILS SYSTEM ASSESSMENT & COMPATIBILITY")
    print("=" * 70)
    
    results = {'tests_passed': 0, 'tests_failed': 0, 'details': []}
    
    # Test 1: Can we import the utils system?
    try:
        from utils.modular_components import ComponentRegistry, ComponentFactory
        from utils.modular_components.registry import _global_registry, register_component
        
        print("✅ Test 1: Utils system import - PASS")
        results['tests_passed'] += 1
        results['details'].append("Utils system successfully imported")
        
    except Exception as e:
        print(f"❌ Test 1: Utils system import - FAIL: {e}")
        results['tests_failed'] += 1
        results['details'].append(f"Import failed: {e}")
        return results
    
    # Test 2: Can we access the global registry?
    try:
        registry = _global_registry
        available_components = registry.list_components()
        
        print("✅ Test 2: Global registry access - PASS")
        print(f"   Available component types: {list(available_components.keys())}")
        for comp_type, components in available_components.items():
            if components:  # Only show non-empty types
                print(f"   {comp_type}: {len(components)} components")
        
        results['tests_passed'] += 1
        results['details'].append(f"Registry accessible with {len(available_components)} component types")
        
    except Exception as e:
        print(f"❌ Test 2: Global registry access - FAIL: {e}")
        results['tests_failed'] += 1
        results['details'].append(f"Registry access failed: {e}")
    
    # Test 3: Can we create a simple component?
    try:
        # Try to create a simple loss component using registry method
        from utils.modular_components.implementations.losses import LossConfig
        
        # Create config object as expected by utils system
        config = LossConfig(reduction='mean')
        
        # Try to get available loss components first
        loss_components = available_components.get('loss', [])
        if loss_components:
            loss_name = loss_components[0]
            mse_loss = registry.create('loss', loss_name, config)
            
            # Test it works
            pred = torch.randn(2, 10, 3, requires_grad=True)
            target = torch.randn(2, 10, 3)
            loss = mse_loss(pred, target)
            loss.backward()
            
            assert pred.grad is not None
            assert torch.isfinite(loss)
            
            print(f"✅ Test 3: Component creation & usage - PASS ({loss_name})")
        else:
            print("⚠️ Test 3: Component creation & usage - SKIPPED (no loss components)")
        
        results['tests_passed'] += 1
        results['details'].append("Component creation system accessible")
        
    except Exception as e:
        print(f"❌ Test 3: Component creation & usage - FAIL: {e}")
        results['tests_failed'] += 1
        results['details'].append(f"Component creation failed: {e}")
    
    # Test 4: What attention components are available?
    try:
        attention_components = available_components.get('attention', [])
        print(f"✅ Test 4: Attention inventory - {len(attention_components)} components found")
        
        if attention_components:
            print("   Available attention components:")
            for comp in attention_components[:5]:  # Show first 5
                print(f"     • {comp}")
            if len(attention_components) > 5:
                print(f"     ... and {len(attention_components) - 5} more")
        
        results['tests_passed'] += 1
        results['details'].append(f"Found {len(attention_components)} attention components")
        
    except Exception as e:
        print(f"❌ Test 4: Attention inventory - FAIL: {e}")
        results['tests_failed'] += 1
        results['details'].append(f"Attention inventory failed: {e}")
    
    # Test 5: Can we create an attention component?
    try:
        if attention_components:
            # Try the first available attention component
            first_attention = attention_components[0]
            
            # Create appropriate config for attention
            from utils.modular_components.config_schemas import AttentionConfig
            config = AttentionConfig(
                d_model=64,
                num_heads=4,
                dropout=0.1
            )
            
            attention = registry.create('attention', first_attention, config)
            
            # Test basic functionality
            x = torch.randn(2, 32, 64)
            output = attention(x, x, x)
            
            print(f"✅ Test 5: Attention creation - PASS ({first_attention})")
            results['tests_passed'] += 1
            results['details'].append(f"Successfully created {first_attention} attention")
            
        else:
            print("⚠️ Test 5: Attention creation - SKIPPED (no attention components)")
            results['details'].append("No attention components available to test")
            
    except Exception as e:
        print(f"❌ Test 5: Attention creation - FAIL: {e}")
        results['tests_failed'] += 1
        results['details'].append(f"Attention creation failed: {e}")
    
    # Test 6: What's the parameter convention?
    try:
        # Check what loss components exist and their conventions
        loss_components = available_components.get('loss', [])
        print(f"✅ Test 6: Loss inventory - {len(loss_components)} components found")
        
        if loss_components:
            print("   Available loss components:")
            for comp in loss_components:
                print(f"     • {comp}")
        
        results['tests_passed'] += 1
        results['details'].append(f"Found {len(loss_components)} loss components")
        
    except Exception as e:
        print(f"❌ Test 6: Loss inventory - FAIL: {e}")
        results['tests_failed'] += 1
        results['details'].append(f"Loss inventory failed: {e}")
    
    # Test 7: Integration potential assessment
    try:
        print("\n🔍 INTEGRATION ASSESSMENT:")
        
        # Check if our components exist in utils
        our_algorithms = ['fourier_attention', 'enhanced_autocorrelation', 'meta_learning_adapter']
        utils_attention = available_components.get('attention', [])
        
        found_in_utils = [alg for alg in our_algorithms if alg in utils_attention]
        missing_in_utils = [alg for alg in our_algorithms if alg not in utils_attention]
        
        print(f"   Our algorithms found in utils: {found_in_utils}")
        print(f"   Our algorithms missing in utils: {missing_in_utils}")
        
        # Check parameter conventions
        print("   Parameter convention check:")
        if loss_components:
            try:
                # Test config object style
                from utils.modular_components.implementations.losses import LossConfig
                config_obj = LossConfig(reduction='mean')
                loss1 = registry.create('loss', loss_components[0], config_obj)
                print("     ✅ Config object works")
            except Exception as e:
                print(f"     ❌ Config object failed: {e}")
        
        results['tests_passed'] += 1
        results['details'].append(f"Integration assessment: {len(found_in_utils)}/{len(our_algorithms)} algorithms found")
        
    except Exception as e:
        print(f"❌ Test 7: Integration assessment - FAIL: {e}")
        results['tests_failed'] += 1
        results['details'].append(f"Integration assessment failed: {e}")
    
    # Summary
    print(f"\n📊 ASSESSMENT SUMMARY:")
    total_tests = results['tests_passed'] + results['tests_failed']
    success_rate = (results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
    
    print(f"   Tests passed: {results['tests_passed']}/{total_tests} ({success_rate:.1f}%)")
    
    if results['tests_failed'] == 0:
        print("   🎉 Utils system fully compatible!")
        print("   ✅ Ready to proceed with algorithm adaptation")
    elif results['tests_passed'] > results['tests_failed']:
        print("   ⚠️ Utils system mostly compatible")
        print("   🔧 Some issues need resolution before proceeding")
    else:
        print("   ❌ Utils system compatibility issues")
        print("   🛠️ Significant work needed for integration")
    
    return results

def test_our_algorithms_compatibility():
    """Test if our algorithms can be adapted to utils/ system"""
    print("\n🧪 ALGORITHM COMPATIBILITY TESTING")
    print("-" * 70)
    
    # Test if we can import our algorithms alongside utils system
    try:
        from layers.modular.attention.fourier_attention import FourierAttention
        from layers.EfficientAutoCorrelation import EfficientAutoCorrelation
        from layers.modular.attention.adaptive_components import MetaLearningAdapter
        from utils.modular_components.base_interfaces import BaseAttention
        
        print("✅ Can import both our algorithms and utils base interfaces")
        
        # Test parameter compatibility
        print("\n🔧 PARAMETER COMPATIBILITY:")
        
        # Our current parameter style
        our_fourier = FourierAttention(d_model=64, n_heads=4)
        print("   ✅ Our algorithms use direct parameter instantiation")
        
        # Utils parameter style would be config-based
        print("   ℹ️ Utils system expects config objects")
        print("   🎯 ADAPTATION NEEDED: Create config wrappers")
        
        return True
        
    except Exception as e:
        print(f"❌ Algorithm compatibility test failed: {e}")
        return False

def main():
    """Main assessment execution"""
    
    # Test utils system capabilities
    utils_results = test_utils_system_capabilities()
    
    # Test our algorithms compatibility
    compat_results = test_our_algorithms_compatibility()
    
    print("\n" + "=" * 70)
    print("🏁 PHASE 1 ASSESSMENT CONCLUSIONS")
    print("=" * 70)
    
    if utils_results['tests_failed'] == 0 and compat_results:
        print("🎉 ASSESSMENT: HIGHLY COMPATIBLE")
        print("✅ Utils system fully functional")
        print("✅ Our algorithms can be adapted")
        print("🚀 RECOMMENDATION: Proceed to Phase 2 (Algorithm Adaptation)")
        
    elif utils_results['tests_passed'] > utils_results['tests_failed'] and compat_results:
        print("⚠️ ASSESSMENT: MOSTLY COMPATIBLE")
        print("✅ Utils system mostly functional")
        print("✅ Our algorithms can be adapted")
        print("🔧 RECOMMENDATION: Fix utils issues, then proceed to Phase 2")
        
    else:
        print("❌ ASSESSMENT: COMPATIBILITY ISSUES")
        print("❌ Significant integration challenges")
        print("🛑 RECOMMENDATION: Consider staying with layers/modular/ system")
    
    print("\nNEXT STEPS:")
    print("1. If compatible: Start Phase 2 (Algorithm Adaptation)")
    print("2. If issues: Resolve compatibility problems first")
    print("3. Create wrapper interfaces for our algorithms")
    print("4. Test wrapped algorithms in utils/ system")

if __name__ == "__main__":
    main()
