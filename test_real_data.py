#!/usr/bin/env python3
"""
Quick runner for REAL financial data systematic tests
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from run_systematic_tests_real_data import main

if __name__ == "__main__":
    print("🚀 Starting REAL Financial Data Systematic Tests...")
    print("   • Using actual 11,181 samples with 113 covariates + 4 targets")
    print("   • seq_len=750, pred_len=20 configuration")
    print("   • Expected to show realistic loss values around 0.1-2.0")
    print()
    
    try:
        results = main()
        print("\n🎉 Real data tests completed successfully!")
        
        # Print key results
        if 'progressive' in results:
            print("\n📊 Key Results:")
            for name, result in results['progressive'].items():
                if result.get('status') == 'success':
                    print(f"   • {result['config_name']}: {result['final_val_loss']:.4f} val loss")
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()