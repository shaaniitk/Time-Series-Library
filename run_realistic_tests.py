#!/usr/bin/env python3
"""
Quick runner for realistic systematic tests
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from run_systematic_tests_realistic import main

if __name__ == "__main__":
    print("🚀 Starting Realistic Systematic Tests...")
    print("   This will test seq_len=750, pred_len=20 configuration")
    print("   Expected to show multi-scale patching provides 40% improvement")
    print()
    
    try:
        results = main()
        print("\n🎉 Tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()