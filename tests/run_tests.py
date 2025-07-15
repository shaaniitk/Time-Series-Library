#!/usr/bin/env python3
"""
Comprehensive test runner for Autoformer models
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_test_suite():
    """Run all tests and provide summary"""
    print("🧪 Running Autoformer Model Test Suite")
    print("=" * 50)
    
    test_files = [
        "test_autoformer_fixed.py",
        "test_enhanced_autoformer_fixed.py", 
        "test_integration.py"
    ]
    
    results = {}
    total_start = time.time()
    
    for test_file in test_files:
        print(f"\n📋 Running {test_file}...")
        start_time = time.time()
        
        try:
            # Run pytest with verbose output
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_file, "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=Path(__file__).parent)
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {test_file} PASSED ({duration:.2f}s)")
                results[test_file] = "PASSED"
            else:
                print(f"❌ {test_file} FAILED ({duration:.2f}s)")
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
                print("STDERR:", result.stderr[-500:])
                results[test_file] = "FAILED"
                
        except Exception as e:
            print(f"💥 {test_file} ERROR: {e}")
            results[test_file] = "ERROR"
    
    total_duration = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    failed = sum(1 for r in results.values() if r == "FAILED")
    errors = sum(1 for r in results.values() if r == "ERROR")
    
    for test_file, result in results.items():
        status_emoji = {"PASSED": "✅", "FAILED": "❌", "ERROR": "💥"}[result]
        print(f"{status_emoji} {test_file}: {result}")
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Duration: {total_duration:.2f}s")
    
    if failed == 0 and errors == 0:
        print("\n🎉 ALL TESTS PASSED! Models are ready for use.")
        return True
    else:
        print(f"\n⚠️  {failed + errors} test(s) failed. Please check the issues above.")
        return False

def quick_smoke_test():
    """Quick smoke test to verify basic functionality"""
    print("🔥 Running Quick Smoke Test...")
    
    try:
        import torch
        from types import SimpleNamespace
        
        # Test AutoformerFixed
        from models.Autoformer_Fixed import Model as AutoformerFixed
        
        config = SimpleNamespace()
        config.task_name = 'long_term_forecast'
        config.seq_len = 96
        config.label_len = 48
        config.pred_len = 24
        config.enc_in = 7
        config.dec_in = 7
        config.c_out = 7
        config.d_model = 64
        config.n_heads = 8
        config.e_layers = 2
        config.d_layers = 1
        config.d_ff = 256
        config.moving_avg = 25
        config.factor = 1
        config.dropout = 0.1
        config.activation = 'gelu'
        config.embed = 'timeF'
        config.freq = 'h'
        config.norm_type = 'LayerNorm'
        
        model = AutoformerFixed(config)
        
        # Quick forward pass
        x_enc = torch.randn(2, 96, 7)
        x_mark_enc = torch.randn(2, 96, 4)
        x_dec = torch.randn(2, 72, 7)
        x_mark_dec = torch.randn(2, 72, 4)
        
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        assert output.shape == (2, 24, 7)
        print("✅ AutoformerFixed smoke test passed")
        
        # Test EnhancedAutoformer
        from models.EnhancedAutoformer_Fixed import EnhancedAutoformer
        
        enhanced_model = EnhancedAutoformer(config)
        
        with torch.no_grad():
            enhanced_output = enhanced_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        assert enhanced_output.shape == (2, 24, 7)
        print("✅ EnhancedAutoformer smoke test passed")
        
        print("🎉 Smoke test completed successfully!")
        return True
        
    except Exception as e:
        print(f"💥 Smoke test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Autoformer Test Suite")
    
    # Run quick smoke test first
    if not quick_smoke_test():
        print("❌ Smoke test failed. Exiting.")
        sys.exit(1)
    
    # Run full test suite
    success = run_test_suite()
    
    if success:
        print("\n🎯 CONCLUSION: Both models are working perfectly!")
        sys.exit(0)
    else:
        print("\n⚠️  CONCLUSION: Some tests failed. Please review and fix.")
        sys.exit(1)