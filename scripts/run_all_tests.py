#!/usr/bin/env python3
"""
Master Test Runner for Modular Autoformer Framework

Runs all test categories in sequence:
1. Individual component tests
2. Integration tests  
3. Factory tests
4. Performance tests
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_script(script_path, description):
    """Run a test script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=False, text=True)
        success = result.returncode == 0
        
        if success:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
        
        return success
        
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run complete test suite."""
    print("="*80)
    print("MODULAR AUTOFORMER FRAMEWORK - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    start_time = time.time()
    scripts_dir = Path(__file__).parent
    
    # Define test sequence
    test_sequence = [
        (scripts_dir / "test_attention_components.py", "Attention Components"),
        (scripts_dir / "test_decomposition_components.py", "Decomposition Components"),
        (scripts_dir / "test_unified_factory.py", "Unified Factory"),
        (scripts_dir / "test_all_components.py", "All Components Integration"),
        (scripts_dir / "test_all_integrations.py", "Full Integration Suite"),
    ]
    
    results = {}
    
    # Run each test script
    for script_path, description in test_sequence:
        if script_path.exists():
            success = run_script(script_path, description)
            results[description] = success
        else:
            print(f"⚠️  {description} - SKIPPED (script not found: {script_path})")
            results[description] = None
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL TEST SUMMARY")
    print('='*80)
    
    passed = sum(1 for success in results.values() if success is True)
    failed = sum(1 for success in results.values() if success is False)
    skipped = sum(1 for success in results.values() if success is None)
    total = len(results)
    
    print(f"Total test suites: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Duration: {duration:.2f} seconds")
    
    print(f"\nDetailed Results:")
    for description, success in results.items():
        if success is True:
            print(f"✅ {description}")
        elif success is False:
            print(f"❌ {description}")
        else:
            print(f"⚠️  {description} (skipped)")
    
    # Overall result
    if failed == 0 and passed > 0:
        print(f"\n🎉 ALL TEST SUITES PASSED! 🎉")
        print("✨ Modular autoformer framework is fully functional! ✨")
        return 0
    elif failed > 0:
        print(f"\n❌ {failed} TEST SUITE(S) FAILED")
        return 1
    else:
        print(f"\n⚠️  NO TESTS WERE RUN")
        return 1

if __name__ == "__main__":
    sys.exit(main())
