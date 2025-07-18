#!/usr/bin/env python3
"""
Convergence test for BayesianEnhancedAutoformer using sanity test mixin
Tests whether the model can converge on a simple synthetic function
"""

import torch
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.models.sanity_test_mixin import SanityTestMixin
from models.BayesianEnhancedAutoformer import BayesianEnhancedAutoformer
from utils.logger import logger

class ConvergenceTest(SanityTestMixin):
    def test_bayesian_convergence(self):
        """Test convergence of BayesianEnhancedAutoformer on synthetic function"""
        
        print("TEST Testing BayesianEnhancedAutoformer Convergence")
        print("=" * 60)
        print("Function: Synthetic sinusoidal patterns")
        print("Target: t1 = sin(X - X1), t2 = sin(X1 - X2), t3 = sin(X2 - X)")
        print("Covariates: cov1 = sin(X), cov2 = sin(X1), cov3 = sin(X2)")
        print("=" * 60)
        
        # Ultra-light config for fast convergence testing
        model_config = {
            'd_model': 32,
            'e_layers': 2,
            'd_layers': 1, 
            'd_ff': 64,
            'n_heads': 4
        }
        
        # Run sanity test with more epochs to see convergence
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        try:
            # Test with 30 epochs since we now have stricter convergence (test MSE < 0.01)
            mse_result, forecast, actual = self.run_sanity_test(
                ModelClass=BayesianEnhancedAutoformer,
                device=device,
                epochs=30,
                model_config=model_config
            )
            
            print("\nTARGET Convergence Test Results:")
            print(f"Final MSE: {mse_result:.6f}")
            
            # Evaluate convergence quality
            if mse_result < 0.01:
                print("PASS EXCELLENT: Model converged very well (MSE < 0.01)")
                convergence_status = "EXCELLENT"
            elif mse_result < 0.1:
                print("PASS GOOD: Model converged reasonably (MSE < 0.1)")
                convergence_status = "GOOD"
            elif mse_result < 1.0:
                print("WARN  MODERATE: Model shows some learning (MSE < 1.0)")
                convergence_status = "MODERATE"
            else:
                print("FAIL POOR: Model did not converge well (MSE >= 1.0)")
                convergence_status = "POOR"
            
            print(f"\nConvergence Status: {convergence_status}")
            print(f"Forecast shape: {forecast.shape}")
            print(f"Actual shape: {actual.shape}")
            
            # Additional analysis
            print("\nCHART Detailed Analysis:")
            for i in range(min(3, forecast.shape[-1])):
                target_mse = ((forecast[:, i] - actual[:, i]) ** 2).mean()
                print(f"Target {i+1} MSE: {target_mse:.6f}")
                
                # Check if this target converged
                if target_mse < 0.01:
                    print(f"  PASS Target {i+1}: Excellent convergence")
                elif target_mse < 0.1:
                    print(f"  PASS Target {i+1}: Good convergence")
                else:
                    print(f"  WARN  Target {i+1}: Poor convergence")
            
            print(f"\nGRAPH Plot saved to: pic/true_forecasting_evaluation.png")
            
            return mse_result, convergence_status
            
        except Exception as e:
            print(f"\n Convergence test failed: {e}")
            import traceback
            traceback.print_exc()
            return None, "FAILED"

def main():
    """Run the convergence test"""
    tester = ConvergenceTest()
    mse, status = tester.test_bayesian_convergence()
    
    print("\n" + "="*60)
    if status == "FAILED":
        print("FAIL CONVERGENCE TEST FAILED")
        return False
    else:
        print(f"PASS CONVERGENCE TEST COMPLETED")
        print(f"Final Result: {status} (MSE: {mse:.6f})")
        
        # Overall assessment
        if status in ["EXCELLENT", "GOOD"]:
            print("PARTY Model demonstrates good learning capability!")
            return True
        else:
            print("WARN  Model may need hyperparameter tuning or more epochs")
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
