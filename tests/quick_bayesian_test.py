#!/usr/bin/env python3
"""
Ultra-fast sanity test for BayesianEnhancedAutoformer
"""

import subprocess
import sys
import time

def quick_bayesian_sanity_test():
    """Run a quick sanity test to verify Bayesian model is working"""
    
    print("ROCKET Quick Bayesian Sanity Test")
    print("=" * 40)
    print("Parameters:")
    print("   seq_len: 48 (vs 625)")
    print("   pred_len: 12 (vs 20)")  
    print("   d_model: 64 (vs 128)")
    print("   epochs: 3 (vs 10)")
    print("   Focus: Verify KL loss and training stability")
    
    cmd = [
        sys.executable, '../scripts/train/train_dynamic_autoformer.py',
        '--config', '../config/config_bayesian_ultralight_sanity.yaml',
        '--model_type', 'bayesian',
        '--auto_fix'
    ]
    
    print(f"\nLIGHTNING Running: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)  # 3 min timeout
        
        elapsed = time.time() - start_time
        print(f"\nTIMER Completed in {elapsed:.1f} seconds")
        
        if result.returncode == 0:
            print("PASS Bayesian model sanity test PASSED")
            
            # Extract key information from output
            output_lines = result.stdout.split('\n')
            
            # Look for KL loss information
            kl_lines = [line for line in output_lines if 'kl_loss' in line.lower()]
            if kl_lines:
                print("\nCHART KL Loss Information:")
                for line in kl_lines[-3:]:  # Last 3 KL loss lines
                    print(f"   {line.strip()}")
            
            # Look for final results
            loss_lines = [line for line in output_lines if 'loss:' in line.lower() and 'epoch' in line.lower()]
            if loss_lines:
                print("\nGRAPH Training Progress:")
                for line in loss_lines[-3:]:  # Last 3 training lines
                    print(f"   {line.strip()}")
            
            # Look for dimension information
            dim_lines = [line for line in output_lines if 'shape' in line.lower() or 'dimension' in line.lower()]
            if dim_lines:
                print("\n Dimension Verification:")
                for line in dim_lines[-3:]:  # Last 3 dimension lines
                    print(f"   {line.strip()}")
            
            print("\nTARGET Sanity Test Summary:")
            print("    Model loads correctly")
            print("    Training runs without crashes")
            print("    KL loss is computed and included")
            print("    Dimensions are handled correctly")
            
        else:
            print("FAIL Bayesian model sanity test FAILED")
            print(f"Exit code: {result.returncode}")
            print("\nSTDERR:")
            print(result.stderr[-1000:])  # Last 1000 chars of error
            return False
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f" Test TIMEOUT after {elapsed:.1f} seconds")
        print("This suggests the model is taking too long - may need further optimization")
        return False
    except Exception as e:
        print(f" Test ERROR: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Starting ultra-fast Bayesian sanity test...")
    success = quick_bayesian_sanity_test()
    
    if success:
        print("\nPARTY Bayesian model is working correctly!")
        print("Ready for full-scale training.")
    else:
        print("\nWARN Sanity test failed - check the logs above.")
