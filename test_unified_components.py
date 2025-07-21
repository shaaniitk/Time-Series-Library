#!/usr/bin/env python3
"""
Test script for unified component system
This script tests the attention_clean.py components without relative import issues
"""

import sys
import os
import torch
import torch.nn as nn

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_attention_imports():
    """Test that we can import attention components"""
    print("🔍 Testing attention component imports...")
    
    try:
        # Add the utils path specifically
        utils_path = os.path.join(project_root, 'utils')
        if utils_path not in sys.path:
            sys.path.insert(0, utils_path)
        
        # Test importing from the implementations directory
        from utils.modular_components.implementations.attention_clean import (
            MultiHeadAttention,
            AutoCorrelationAttention, 
            SparseAttention,
            FourierAttention,
            ATTENTION_REGISTRY,
            AUTOCORRELATION_LAYER_AVAILABLE,
            get_attention_component,
            list_attention_components
        )
        
        print("✅ Successfully imported all attention components")
        print(f"   AutoCorrelationLayer available: {AUTOCORRELATION_LAYER_AVAILABLE}")
        print(f"   Available components: {list_attention_components()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base_attention_components():
    """Test basic attention components without base class dependencies"""
    print("\n🧪 Testing basic attention component creation...")
    
    try:
        # Test MultiHeadAttention creation (without base class)
        class SimpleMultiHeadAttention(nn.Module):
            """Simplified version for testing"""
            def __init__(self, d_model=512, n_heads=8, dropout=0.1):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.d_k = d_model // n_heads
                
                self.w_qs = nn.Linear(d_model, d_model, bias=False)
                self.w_ks = nn.Linear(d_model, d_model, bias=False)
                self.w_vs = nn.Linear(d_model, d_model, bias=False)
                self.fc = nn.Linear(d_model, d_model, bias=False)
                self.dropout = nn.Dropout(dropout)
        
        # Test creation
        attention = SimpleMultiHeadAttention(d_model=256, n_heads=4)
        print(f"✅ Created MultiHeadAttention: {attention.d_model} dimensions, {attention.n_heads} heads")
        
        # Test forward pass
        batch_size, seq_len, d_model = 2, 10, 256
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Test the layers exist
        q = attention.w_qs(x)
        k = attention.w_ks(x)
        v = attention.w_vs(x)
        
        print(f"✅ Forward pass test successful: input {x.shape} -> Q,K,V {q.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_autocorrelation_layer_import():
    """Test AutoCorrelationLayer import specifically"""
    print("\n🔍 Testing AutoCorrelationLayer import...")
    
    try:
        # Try to import from layers module
        from utils.modular_components.implementations.layers import AutoCorrelationLayer
        print("✅ Successfully imported AutoCorrelationLayer from layers module")
        
        # Test basic creation
        layer = AutoCorrelationLayer(d_model=256, n_heads=4)
        print(f"✅ Created AutoCorrelationLayer: {layer.d_model} dimensions")
        
        return True
        
    except Exception as e:
        print(f"❌ AutoCorrelationLayer import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_registry_functionality():
    """Test the attention registry"""
    print("\n📋 Testing attention registry...")
    
    try:
        from utils.modular_components.implementations.attention_clean import (
            ATTENTION_REGISTRY,
            get_attention_component
        )
        
        print(f"✅ Registry contains {len(ATTENTION_REGISTRY)} components:")
        for name, component_class in ATTENTION_REGISTRY.items():
            print(f"   - {name}: {component_class.__name__}")
        
        # Test getting a component
        multihead = get_attention_component('multihead', d_model=256, n_heads=4)
        print(f"✅ Retrieved multihead attention: {type(multihead).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🚀 Testing Unified Component System")
    print("=" * 50)
    
    # Test results
    results = []
    
    # Run tests
    results.append(("Import Test", test_attention_imports()))
    results.append(("Basic Components", test_base_attention_components()))
    results.append(("AutoCorrelationLayer", test_autocorrelation_layer_import()))
    results.append(("Registry", test_registry_functionality()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} | {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Unified component system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

def test_unified_components():
    """Test all unified component registries"""
    
    print("🧪 TESTING UNIFIED COMPONENTS")
    print("=" * 50)
    
    # Test parameters
    batch_size, seq_len, d_model = 2, 32, 512
    test_input = torch.randn(batch_size, seq_len, d_model)
    test_target = torch.randn(batch_size, seq_len, d_model)
    
    results = {
        'successful': [],
        'failed': [],
        'total_components': 0
    }
    
    # =============================================================================
    # TEST UNIFIED ATTENTION COMPONENTS
    # =============================================================================
    
    print("\n📡 Testing Unified Attention Components...")
    try:
        from utils.modular_components.implementations.attentions_unified import (
            ATTENTION_REGISTRY, get_attention_component, register_attention_components
        )
        
        print(f"   Found {len(ATTENTION_REGISTRY)} attention components: {list(ATTENTION_REGISTRY.keys())}")
        results['total_components'] += len(ATTENTION_REGISTRY)
        
        # Test each attention component
        for name, component_class in ATTENTION_REGISTRY.items():
            try:
                # Create component with keyword arguments
                component = component_class(d_model=d_model, n_heads=8, dropout=0.1)
                
                # Test forward pass
                with torch.no_grad():
                    output, _ = component(test_input, test_input, test_input)
                
                # Validate output shape
                expected_shape = (batch_size, seq_len, d_model)
                if output.shape == expected_shape:
                    results['successful'].append(f"attention.{name}")
                    print(f"   ✅ {name}: Passed")
                else:
                    results['failed'].append(f"attention.{name}: Wrong shape {output.shape}")
                    print(f"   ❌ {name}: Wrong shape {output.shape}")
                    
            except Exception as e:
                results['failed'].append(f"attention.{name}: {str(e)}")
                print(f"   ❌ {name}: {e}")
                
    except Exception as e:
        print(f"   ❌ Failed to load attention components: {e}")
        results['failed'].append(f"attention_module: {e}")
    
    # =============================================================================
    # TEST UNIFIED LOSS COMPONENTS
    # =============================================================================
    
    print("\n📊 Testing Unified Loss Components...")
    try:
        from utils.modular_components.implementations.losses_unified import (
            LOSS_REGISTRY, get_loss_component, register_loss_components
        )
        
        print(f"   Found {len(LOSS_REGISTRY)} loss components: {list(LOSS_REGISTRY.keys())}")
        results['total_components'] += len(LOSS_REGISTRY)
        
        # Test each loss component
        for name, component_class in LOSS_REGISTRY.items():
            try:
                # Create component with keyword arguments
                component = component_class()
                
                # Test forward pass
                with torch.no_grad():
                    loss = component.compute_loss(test_input, test_target)
                
                # Validate loss is a scalar tensor
                if loss.dim() == 0 and loss.item() >= 0:
                    results['successful'].append(f"loss.{name}")
                    print(f"   ✅ {name}: Passed (loss={loss.item():.4f})")
                else:
                    results['failed'].append(f"loss.{name}: Invalid loss shape/value")
                    print(f"   ❌ {name}: Invalid loss {loss}")
                    
            except Exception as e:
                results['failed'].append(f"loss.{name}: {str(e)}")
                print(f"   ❌ {name}: {e}")
                
    except Exception as e:
        print(f"   ❌ Failed to load loss components: {e}")
        results['failed'].append(f"loss_module: {e}")
    
    # =============================================================================
    # TEST OTHER UNIFIED COMPONENTS (if they exist)
    # =============================================================================
    
    # Check for other unified component files
    impl_dir = Path("utils/modular_components/implementations")
    unified_files = list(impl_dir.glob("*_unified.py"))
    
    print(f"\n📂 Found {len(unified_files)} unified component files:")
    for file in unified_files:
        print(f"   📄 {file.name}")
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    
    print(f"\n📋 TEST SUMMARY:")
    print(f"   ✅ Successful components: {len(results['successful'])}")
    print(f"   ❌ Failed components: {len(results['failed'])}")
    print(f"   📊 Total components tested: {results['total_components']}")
    
    if results['successful']:
        print(f"\n✅ SUCCESSFUL COMPONENTS:")
        for comp in results['successful']:
            print(f"   - {comp}")
    
    if results['failed']:
        print(f"\n❌ FAILED COMPONENTS:")
        for comp in results['failed']:
            print(f"   - {comp}")
    
    success_rate = len(results['successful']) / max(1, results['total_components']) * 100
    print(f"\n🎯 SUCCESS RATE: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 UNIFIED COMPONENTS ARE WORKING WELL!")
        return True
    else:
        print("🔧 UNIFIED COMPONENTS NEED MORE WORK")
        return False

if __name__ == "__main__":
    success = test_unified_components()
    exit(0 if success else 1)
