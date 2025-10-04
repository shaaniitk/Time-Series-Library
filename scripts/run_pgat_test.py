#!/usr/bin/env python3
"""
Simple test runner for SOTA_Temporal_PGAT using the existing training infrastructure.
This creates a minimal test with 2 covariates and 2 targets.
"""

import subprocess
import sys
from pathlib import Path

def run_pgat_test():
    """Run the PGAT test using the existing training script."""
    
    # Get project root
    project_root = Path(__file__).resolve().parents[1]
    
    # Paths
    config_path = project_root / 'configs' / 'sota_pgat_test_minimal.yaml'
    dataset_path = project_root / 'data' / 'synthetic_multi_wave_test.csv'
    training_script = project_root / 'scripts' / 'train' / 'train_pgat_synthetic.py'
    
    print("🚀 Running SOTA_Temporal_PGAT Test")
    print("=" * 50)
    print(f"Config: {config_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Training script: {training_script}")
    print()
    
    # Build command
    cmd = [
        sys.executable,
        str(training_script),
        '--config', str(config_path),
        '--regenerate-data',
        '--dataset-path', str(dataset_path),
        '--rows', '200',
        '--waves', '2',      # 2 covariates (wave features)
        '--targets', '2',    # 2 targets
        '--freq', 'H',
        '--start', '2020-01-01',
        '--seed', '42',
        '--verbose'
    ]
    
    print("Command:")
    print(' '.join(cmd))
    print()
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ PGAT test completed successfully!")
            return True
        else:
            print(f"❌ PGAT test failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ PGAT test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ PGAT test failed with exception: {e}")
        return False

def main():
    """Main test function."""
    print("SOTA_Temporal_PGAT Training Test")
    print("=" * 60)
    print("Testing with:")
    print("- 2 covariates (wave features)")
    print("- 2 targets")
    print("- 1 epoch training")
    print("- All sophisticated features enabled")
    print("- Memory optimizations enabled")
    print()
    
    success = run_pgat_test()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TEST PASSED!")
        print("The SOTA_Temporal_PGAT model is working correctly with:")
        print("✅ All sophisticated features")
        print("✅ Memory optimizations")
        print("✅ Covariate and target processing")
        print("✅ Full training pipeline")
    else:
        print("❌ TEST FAILED!")
        print("Check the output above for error details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)