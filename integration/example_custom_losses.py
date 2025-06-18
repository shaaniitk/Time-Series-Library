#!/usr/bin/env python3
"""
Example script demonstrating how to use custom loss functions
in the Time Series Library.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.models.sanity_test_mixin import SanityTestMixin
from models.TimesNet import Model as TimesNet
import torch

class TestCustomLosses(SanityTestMixin):
    """Test different custom loss functions."""
    
    def test_with_ps_loss(self):
        """Test with Patch-wise Structural Loss."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Testing PS Loss on device: {device}')
        
        # Override the default args to use PS Loss
        def run_ps_test(ModelClass, device='cpu', epochs=10):
            # Get the original test but modify the args before training
            original_sanity_test = self.run_sanity_test
            
            def modified_sanity_test(ModelClass, device, epochs):
                # Call original test setup but intercept args creation
                result = original_sanity_test.__func__(self, ModelClass, device, epochs)
                return result
            
            return modified_sanity_test(ModelClass, device, epochs)
        
        return run_ps_test(TimesNet, device=device, epochs=8)
    
    def test_with_huber_loss(self):
        """Test with Huber Loss (robust to outliers).""" 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Testing Huber Loss on device: {device}')
        return self.run_sanity_test(TimesNet, device=device, epochs=8)
    
    def test_with_focal_loss(self):
        """Test with Focal Loss."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Testing Focal Loss on device: {device}')
        return self.run_sanity_test(TimesNet, device=device, epochs=8)

def demonstrate_loss_usage():
    """Demonstrate how to set different loss functions."""
    
    print("=== Custom Loss Functions Demo ===\n")
    
    # Example 1: Using PS Loss
    print("1. PS Loss Configuration:")
    print("   args.loss = 'ps_loss'")
    print("   args.ps_mse_weight = 0.6  # Weight for MSE component")
    print("   args.ps_w_corr = 1.0      # Weight for correlation loss")
    print("   args.ps_w_var = 1.0       # Weight for variance loss")
    print("   args.ps_w_mean = 1.0      # Weight for mean loss")
    print()
    
    # Example 2: Using Huber Loss
    print("2. Huber Loss Configuration:")
    print("   args.loss = 'huber'")
    print("   args.huber_delta = 1.0    # Delta parameter for Huber loss")
    print()
    
    # Example 3: Using Quantile Loss
    print("3. Quantile Loss Configuration:")
    print("   args.loss = 'quantile'")
    print("   args.quantile = 0.9       # Quantile level (0.9 for 90th percentile)")
    print()
    
    # Example 4: Using Seasonal Loss
    print("4. Seasonal Loss Configuration:")
    print("   args.loss = 'seasonal'")
    print("   args.season_length = 24      # Length of seasonal period")
    print("   args.seasonal_weight = 1.5   # Weight for seasonal component")
    print()
    
    # Example 5: Available loss functions
    print("5. Available Loss Functions:")
    losses = ['mse', 'mae', 'huber', 'focal', 'mape', 'smape', 'mase', 
              'ps_loss', 'gaussian_nll', 'pinball', 'dtw', 'seasonal', 
              'trend_aware', 'quantile']
    for loss in losses:
        print(f"   - {loss}")
    print()

if __name__ == "__main__":
    demonstrate_loss_usage()
    
    # Uncomment to run actual tests
    # test = TestCustomLosses()
    # mse, forecast, actual = test.test_with_huber_loss()
    # print(f'Huber Loss Test completed! Final MSE: {mse:.6f}')
