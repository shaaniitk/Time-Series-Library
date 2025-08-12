#!/usr/bin/env python3
"""
Focused Core Component Testing

Tests only the core working components that we know should pass.
This establishes a baseline for what's actually working in Phase 1 and Phase 2.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_core_working_components():
    """Test only the core components we know work"""
    print("🎯 CORE COMPONENT VALIDATION")
    print("=" * 60)
    
    results = {'passed': 0, 'failed': 0}
    
    print("\n📊 PHASE 1: Core Loss Components")
    print("-" * 40)
    
    # Test basic loss components that should work
    try:
        from layers.modular.losses.advanced_losses import MAPELoss, SMAPELoss, FocalLoss
        
        # Test MAPE
        mape = MAPELoss()
        pred = torch.randn(2, 10, 3, requires_grad=True)
        target = torch.abs(torch.randn(2, 10, 3)) + 0.1  # Ensure positive for MAPE
        loss = mape(pred, target)
        loss.backward()
        assert pred.grad is not None
        print("✅ MAPELoss: PASS")
        results['passed'] += 1
        
        # Test SMAPE
        smape = SMAPELoss()
        pred = torch.randn(2, 10, 3, requires_grad=True)
        target = torch.abs(torch.randn(2, 10, 3)) + 0.1
        loss = smape(pred, target)
        loss.backward()
        assert pred.grad is not None
        print("✅ SMAPELoss: PASS")
        results['passed'] += 1
        
        # Test Focal
        focal = FocalLoss()
        pred = torch.randn(2, 10, 3, requires_grad=True)
        target = torch.randn(2, 10, 3)
        loss = focal(pred, target)
        loss.backward()
        assert pred.grad is not None
        print("✅ FocalLoss: PASS")
        results['passed'] += 1
        
    except Exception as e:
        print(f"❌ Advanced Losses: FAIL - {e}")
        results['failed'] += 1
    
    print("\n🎯 PHASE 2: Core Attention Components")
    print("-" * 40)
    
    # Test core attention components that should work
    
    # Test 1: FourierAttention (restored algorithm)
    try:
        from layers.modular.attention.fourier_attention import FourierAttention
        attn = FourierAttention(d_model=64, n_heads=4)
        
        queries = torch.randn(2, 32, 64, requires_grad=True)
        keys = torch.randn(2, 32, 64, requires_grad=True)
        values = torch.randn(2, 32, 64, requires_grad=True)
        
        output, _ = attn(queries, keys, values)
        assert output.shape == queries.shape
        
        # Test gradients
        loss = output.sum()
        loss.backward()
        assert queries.grad is not None
        print("✅ FourierAttention: PASS")
        results['passed'] += 1
        
    except Exception as e:
        print(f"❌ FourierAttention: FAIL - {e}")
        results['failed'] += 1
    
    # Test 2: WaveletAttention  
    try:
        from layers.modular.attention.wavelet_attention import WaveletAttention
        attn = WaveletAttention(d_model=64, n_heads=4)
        
        queries = torch.randn(2, 32, 64, requires_grad=True)
        keys = torch.randn(2, 32, 64, requires_grad=True)
        values = torch.randn(2, 32, 64, requires_grad=True)
        
        output, _ = attn(queries, keys, values)
        assert output.shape == queries.shape
        
        # Test gradients
        loss = output.sum()
        loss.backward()
        assert queries.grad is not None
        print("✅ WaveletAttention: PASS")
        results['passed'] += 1
        
    except Exception as e:
        print(f"❌ WaveletAttention: FAIL - {e}")
        results['failed'] += 1
    
    # Test 3: Enhanced AutoCorrelation (restored algorithm)
    try:
        from layers.EfficientAutoCorrelation import EfficientAutoCorrelation
        autocorr = EfficientAutoCorrelation(adaptive_k=True, multi_scale=True)
        
        queries = torch.randn(2, 32, 4, 16, requires_grad=True)
        keys = torch.randn(2, 32, 4, 16, requires_grad=True)
        values = torch.randn(2, 32, 4, 16, requires_grad=True)
        
        output, _ = autocorr(queries, keys, values, None)
        assert output.shape == queries.shape
        
        # Verify sophisticated k_predictor exists
        assert hasattr(autocorr, 'k_predictor')
        assert len(list(autocorr.k_predictor.modules())) > 1
        print("✅ Enhanced AutoCorrelation: PASS")
        results['passed'] += 1
        
    except Exception as e:
        print(f"❌ Enhanced AutoCorrelation: FAIL - {e}")
        results['failed'] += 1
    
    # Test 4: MetaLearning Adapter (restored algorithm)
    try:
        from layers.modular.attention.adaptive_components import MetaLearningAdapter
        meta_attn = MetaLearningAdapter(d_model=64, n_heads=4)
        
        queries = torch.randn(2, 32, 64, requires_grad=True)
        keys = torch.randn(2, 32, 64, requires_grad=True)
        values = torch.randn(2, 32, 64, requires_grad=True)
        
        output, _ = meta_attn(queries, keys, values)
        assert output.shape == queries.shape
        
        # Verify MAML components exist
        assert hasattr(meta_attn, 'fast_weights')
        print("✅ MetaLearning Adapter: PASS")
        results['passed'] += 1
        
    except Exception as e:
        print(f"❌ MetaLearning Adapter: FAIL - {e}")
        results['failed'] += 1
    
    # Test 5: BayesianAttention
    try:
        from layers.modular.attention.bayesian_attention import BayesianAttention
        attn = BayesianAttention(d_model=64, n_heads=4)
        
        queries = torch.randn(2, 32, 64, requires_grad=True)
        keys = torch.randn(2, 32, 64, requires_grad=True)
        values = torch.randn(2, 32, 64, requires_grad=True)
        
        output, _ = attn(queries, keys, values)
        assert output.shape == queries.shape
        
        # Test gradients
        loss = output.sum()
        loss.backward()
        assert queries.grad is not None
        print("✅ BayesianAttention: PASS")
        results['passed'] += 1
        
    except Exception as e:
        print(f"❌ BayesianAttention: FAIL - {e}")
        results['failed'] += 1
    
    print("\n🔗 INTEGRATION: Core Component Integration")
    print("-" * 40)
    
    # Test integration of working components
    try:
        from layers.modular.losses.advanced_losses import MAPELoss
        from layers.modular.attention.fourier_attention import FourierAttention
        
        # Create components
        loss_fn = MAPELoss()
        attn = FourierAttention(d_model=64, n_heads=4)
        
        # Simulate training step
        x = torch.randn(2, 24, 64, requires_grad=True)
        target = torch.abs(torch.randn(2, 24, 64)) + 0.1
        
        # Forward pass
        attn_out, _ = attn(x, x, x)
        loss = loss_fn(attn_out, target)
        
        # Backward pass
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(loss)
        
        print("✅ Core Integration: PASS")
        results['passed'] += 1
        
    except Exception as e:
        print(f"❌ Core Integration: FAIL - {e}")
        results['failed'] += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏆 CORE COMPONENT VALIDATION RESULTS")
    print("=" * 60)
    
    total = results['passed'] + results['failed']
    if total > 0:
        success_rate = (results['passed'] / total) * 100
        print(f"CORE COMPONENTS   | {results['passed']:2d}/{total:2d} | {success_rate:5.1f}% | ", end="")
        
        if results['failed'] == 0:
            print("🎉 SUCCESS")
            status = "✅ All core components working!"
        elif results['passed'] > results['failed']:
            print("⚠️  PARTIAL")
            status = "⚠️ Most core components working, some issues"
        else:
            print("❌ FAILURE")
            status = "❌ Major issues with core components"
    else:
        print("NO TESTS RUN")
        status = "❓ No components tested"
    
    print("=" * 60)
    print(f"Status: {status}")
    
    if results['failed'] == 0:
        print("\n💡 RECOMMENDATIONS:")
        print("🎯 Core functionality solid!")
        print("🚀 Ready to expand component coverage") 
        print("📈 Can proceed with confidence to Phase 3")
    else:
        print("\n💡 RECOMMENDATIONS:")
        print("🔧 Fix core component issues first")
        print("🧪 Re-test after fixes")
        print("⏸️  Hold Phase 3 until core is stable")
    
    return results['failed'] == 0

if __name__ == "__main__":
    success = test_core_working_components()
    sys.exit(0 if success else 1)
