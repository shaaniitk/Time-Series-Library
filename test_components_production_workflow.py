#!/usr/bin/env python3
"""
Quick runner for Production Workflow Component Testing
Uses the exact same workflow as train_celestial_production.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from run_systematic_tests_production_workflow import main

if __name__ == "__main__":
    print("🚀 Starting Production Workflow Component Testing...")
    print("   • Uses EXACT same workflow as train_celestial_production.py")
    print("   • Proper data scaling and matrix operations")
    print("   • Lightweight configs for fast component comparison")
    print("   • No negative losses - realistic validation")
    print()
    
    try:
        results = main()
        print("\n🎉 Production workflow tests completed successfully!")
        
        # Print summary of successful tests
        successful_tests = []
        for test_type, test_results in results.items():
            if isinstance(test_results, dict):
                for name, result in test_results.items():
                    if result.get('status') == 'success':
                        val_loss = result.get('final_val_loss', 'N/A')
                        successful_tests.append(f"   • {name}: {val_loss:.6f} val loss")
        
        if successful_tests:
            print("\n📊 Successful Tests:")
            for test in successful_tests[:10]:  # Show first 10
                print(test)
            if len(successful_tests) > 10:
                print(f"   ... and {len(successful_tests) - 10} more")
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()