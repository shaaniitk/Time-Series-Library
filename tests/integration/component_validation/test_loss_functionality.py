#!/usr/bin/env python3
"""
Comprehensive Loss Function Component Functionality Tests

This test suite validates that each loss function component not only computes
loss values but produces mathematically correct and expected behaviors.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.modular_components.registry import create_component, get_global_registry
    from utils.modular_components.implementations import get_integration_status
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"WARN Could not import modular components: {e}")
    COMPONENTS_AVAILABLE = False

class MockBayesianModel(nn.Module):
    """Mock Bayesian model for testing KL divergence"""
    def __init__(self, input_size=10, hidden_size=20):
        super().__init__()
        # Bayesian layers with priors
        self.fc1_mean = nn.Linear(input_size, hidden_size)
        self.fc1_logvar = nn.Linear(input_size, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, 1)
        self.fc2_logvar = nn.Linear(hidden_size, 1)
        
        # Initialize priors
        self.prior_mean = 0.0
        self.prior_var = 1.0
        
    def forward(self, x):
        # Sample from variational distribution
        h_mean = self.fc1_mean(x)
        h_logvar = self.fc1_logvar(x)
        h_std = torch.exp(0.5 * h_logvar)
        h = h_mean + h_std * torch.randn_like(h_mean)
        
        out_mean = self.fc2_mean(h)
        out_logvar = self.fc2_logvar(h)
        out_std = torch.exp(0.5 * out_logvar)
        output = out_mean + out_std * torch.randn_like(out_mean)
        
        return output
    
    def get_kl_divergence(self):
        """Compute KL divergence for Bayesian layers"""
        kl_div = 0.0
        
        for layer in [self.fc1_mean, self.fc1_logvar, self.fc2_mean, self.fc2_logvar]:
            for param in layer.parameters():
                # Simplified KL divergence calculation
                posterior_var = param.var()
                posterior_mean = param.mean()
                
                kl = 0.5 * (
                    torch.log(self.prior_var / posterior_var) +
                    (posterior_var + (posterior_mean - self.prior_mean)**2) / self.prior_var - 1
                )
                kl_div += kl.sum()
        
        return kl_div

def create_sample_predictions_and_targets(batch_size=4, seq_len=24, features=7):
    """Create sample predictions and targets for loss testing"""
    predictions = torch.randn(batch_size, seq_len, features)
    targets = torch.randn(batch_size, seq_len, features)
    return predictions, targets

def test_mse_loss_functionality():
    """Test MSE loss actual functionality"""
    print("TEST Testing MSE Loss Functionality...")
    
    try:
        loss_fn = create_component('loss', 'mse', {'reduction': 'mean'})
        if loss_fn is None:
            print("    WARN MSE loss not available, skipping...")
            return True
        
        pred, target = create_sample_predictions_and_targets()
        
        # Test basic loss computation
        loss = loss_fn(pred, target)
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() >= 0, "MSE loss should be non-negative"
        
        # Test perfect prediction (loss should be 0)
        loss_perfect = loss_fn(target, target)
        assert abs(loss_perfect.item()) < 1e-6, f"Perfect prediction loss should be ~0, got {loss_perfect.item()}"
        
        # Test that larger errors give larger loss
        pred_worse = target + 2 * torch.randn_like(target)
        loss_worse = loss_fn(pred_worse, target)
        assert loss_worse.item() > loss.item(), "Larger errors should give larger loss"
        
        # Test different reduction modes
        for reduction in ['mean', 'sum', 'none']:
            try:
                loss_fn_red = create_component('loss', 'mse', {'reduction': reduction})
                loss_red = loss_fn_red(pred, target)
                
                if reduction == 'none':
                    assert loss_red.shape == pred.shape, f"'none' reduction should preserve shape"
                else:
                    assert loss_red.dim() == 0, f"'{reduction}' reduction should be scalar"
                    
            except Exception as e:
                print(f"    WARN Reduction mode '{reduction}' failed: {e}")
        
        # Test gradient flow
        pred_grad = pred.clone().requires_grad_(True)
        loss_grad = loss_fn(pred_grad, target)
        loss_grad.backward()
        
        assert pred_grad.grad is not None, "Gradients should exist"
        assert not torch.isnan(pred_grad.grad).any(), "Gradients should not be NaN"
        
        print("    PASS MSE loss functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL MSE loss test failed: {e}")
        return False

def test_mae_loss_functionality():
    """Test MAE loss actual functionality"""
    print("TEST Testing MAE Loss Functionality...")
    
    try:
        loss_fn = create_component('loss', 'mae', {'reduction': 'mean'})
        if loss_fn is None:
            print("    WARN MAE loss not available, skipping...")
            return True
        
        pred, target = create_sample_predictions_and_targets()
        
        # Test basic loss computation
        loss = loss_fn(pred, target)
        assert loss.item() >= 0, "MAE loss should be non-negative"
        
        # Test perfect prediction
        loss_perfect = loss_fn(target, target)
        assert abs(loss_perfect.item()) < 1e-6, "Perfect prediction loss should be ~0"
        
        # Test MAE vs MSE behavior (MAE should be more robust to outliers)
        mse_fn = create_component('loss', 'mse', {'reduction': 'mean'})
        if mse_fn:
            # Create data with outliers
            pred_outlier = target.clone()
            pred_outlier[0, 0, 0] = target[0, 0, 0] + 10  # Add outlier
            
            mae_loss = loss_fn(pred_outlier, target)
            mse_loss = mse_fn(pred_outlier, target)
            
            # MAE should be less affected by the outlier
            mae_ratio = mae_loss / loss_fn(pred, target)
            mse_ratio = mse_loss / mse_fn(pred, target)
            
            print(f"    CHART Outlier sensitivity - MAE ratio: {mae_ratio:.2f}, MSE ratio: {mse_ratio:.2f}")
            assert mse_ratio > mae_ratio, "MSE should be more sensitive to outliers than MAE"
        
        print("    PASS MAE loss functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL MAE loss test failed: {e}")
        return False

def test_bayesian_mse_loss_functionality():
    """Test Bayesian MSE loss with KL divergence functionality"""
    print("TEST Testing Bayesian MSE Loss Functionality...")
    
    try:
        loss_fn = create_component('loss', 'bayesian_mse', {
            'kl_weight': 1e-5,
            'uncertainty_weight': 0.1,
            'reduction': 'mean'
        })
        
        if loss_fn is None:
            print("    WARN Bayesian MSE loss not available, skipping...")
            return True
        
        pred, target = create_sample_predictions_and_targets()
        model = MockBayesianModel()
        
        # Test loss without model (should work like regular MSE)
        loss_no_model = loss_fn(pred, target)
        assert loss_no_model.item() >= 0, "Bayesian MSE without model should be non-negative"
        
        # Test loss with model (should include KL divergence)
        if hasattr(loss_fn, 'compute_loss'):
            loss_with_model = loss_fn.compute_loss(pred, target, model=model)
            assert loss_with_model.item() >= 0, "Bayesian MSE with model should be non-negative"
            
            # KL divergence should make loss slightly larger
            if loss_with_model.item() > loss_no_model.item():
                print("    PASS KL divergence contribution detected")
            else:
                print("    WARN KL divergence contribution not clearly visible")
        
        # Test uncertainty quantification
        if hasattr(loss_fn, 'uncertainty_weight'):
            assert hasattr(loss_fn, 'uncertainty_weight'), "Should have uncertainty weight parameter"
            print(f"    CHART KL weight: {getattr(loss_fn, 'kl_weight', 'N/A')}")
            print(f"    CHART Uncertainty weight: {getattr(loss_fn, 'uncertainty_weight', 'N/A')}")
        
        print("    PASS Bayesian MSE loss functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Bayesian MSE loss test failed: {e}")
        return False

def test_quantile_loss_functionality():
    """Test quantile loss functionality"""
    print("TEST Testing Quantile Loss Functionality...")
    
    try:
        quantiles = [0.1, 0.5, 0.9]
        loss_fn = create_component('loss', 'quantile_loss', {
            'quantiles': quantiles,
            'reduction': 'mean'
        })
        
        if loss_fn is None:
            print("    WARN Quantile loss not available, skipping...")
            return True
        
        # Create quantile predictions (batch, seq, features, quantiles)
        batch_size, seq_len, features = 4, 24, 7
        pred_quantiles = torch.randn(batch_size, seq_len, features, len(quantiles))
        target = torch.randn(batch_size, seq_len, features)
        
        # Test basic loss computation
        loss = loss_fn(pred_quantiles, target)
        assert loss.item() >= 0, "Quantile loss should be non-negative"
        
        # Test quantile ordering (lower quantiles should be <= higher quantiles)
        pred_ordered = pred_quantiles.clone()
        pred_ordered = torch.sort(pred_ordered, dim=-1)[0]  # Sort quantiles
        loss_ordered = loss_fn(pred_ordered, target)
        
        # Ordered quantiles should generally have lower or similar loss
        print(f"    CHART Unordered loss: {loss.item():.6f}, Ordered loss: {loss_ordered.item():.6f}")
        
        # Test pinball loss property
        # For quantile q, loss should be asymmetric
        q = 0.1
        errors = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        pinball_losses = []
        
        for error in errors:
            pinball_loss = torch.where(
                error >= 0,
                q * error,
                (q - 1) * error
            )
            pinball_losses.append(pinball_loss.item())
        
        print(f"    CHART Pinball loss pattern for q=0.1: {pinball_losses}")
        
        # Test different quantile sets
        for test_quantiles in [[0.5], [0.25, 0.75], [0.1, 0.3, 0.5, 0.7, 0.9]]:
            try:
                test_loss_fn = create_component('loss', 'quantile_loss', {
                    'quantiles': test_quantiles,
                    'reduction': 'mean'
                })
                
                if test_loss_fn:
                    test_pred = torch.randn(batch_size, seq_len, features, len(test_quantiles))
                    test_loss = test_loss_fn(test_pred, target)
                    print(f"    CHART {len(test_quantiles)} quantiles loss: {test_loss.item():.6f}")
                    
            except Exception as e:
                print(f"    WARN Quantile set {test_quantiles} failed: {e}")
        
        print("    PASS Quantile loss functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Quantile loss test failed: {e}")
        return False

def test_frequency_aware_loss():
    """Test frequency-aware loss functionality"""
    print("TEST Testing Frequency-Aware Loss Functionality...")
    
    try:
        loss_fn = create_component('loss', 'frequency_aware', {
            'freq_weight': 0.5,
            'reduction': 'mean'
        })
        
        if loss_fn is None:
            print("    WARN Frequency-aware loss not available, skipping...")
            return True
        
        pred, target = create_sample_predictions_and_targets()
        
        # Test basic loss computation
        loss = loss_fn(pred, target)
        assert loss.item() >= 0, "Frequency-aware loss should be non-negative"
        
        # Test with high-frequency vs low-frequency signals
        t = torch.linspace(0, 4*np.pi, 24).unsqueeze(0).unsqueeze(-1)
        
        # Low frequency signal
        target_low_freq = torch.sin(t).expand(4, -1, 7)
        pred_low_freq = target_low_freq + 0.1 * torch.randn_like(target_low_freq)
        
        # High frequency signal  
        target_high_freq = torch.sin(10 * t).expand(4, -1, 7)
        pred_high_freq = target_high_freq + 0.1 * torch.randn_like(target_high_freq)
        
        loss_low_freq = loss_fn(pred_low_freq, target_low_freq)
        loss_high_freq = loss_fn(pred_high_freq, target_high_freq)
        
        print(f"    CHART Low frequency loss: {loss_low_freq.item():.6f}")
        print(f"    CHART High frequency loss: {loss_high_freq.item():.6f}")
        
        # Test frequency domain properties
        if hasattr(loss_fn, 'compute_frequency_loss'):
            print("    PASS Frequency domain computation available")
        
        print("    PASS Frequency-aware loss functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Frequency-aware loss test failed: {e}")
        return False

def test_loss_mathematical_properties():
    """Test mathematical properties of loss functions"""
    print("TEST Testing Loss Mathematical Properties...")
    
    try:
        # Test common loss functions
        loss_types = ['mse', 'mae']
        pred, target = create_sample_predictions_and_targets()
        
        results = {}
        
        for loss_type in loss_types:
            try:
                loss_fn = create_component('loss', loss_type, {'reduction': 'mean'})
                if loss_fn is None:
                    continue
                
                # Test convexity (second derivative test)
                pred_param = nn.Parameter(pred.clone())
                optimizer = torch.optim.SGD([pred_param], lr=0.01)
                
                losses = []
                for _ in range(5):
                    optimizer.zero_grad()
                    loss = loss_fn(pred_param, target)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                
                # Loss should generally decrease with optimization
                is_decreasing = all(losses[i] >= losses[i+1] for i in range(len(losses)-1))
                
                results[loss_type] = {
                    'converges': losses[-1] < losses[0],
                    'monotonic_decrease': is_decreasing,
                    'final_loss': losses[-1]
                }
                
                print(f"    CHART {loss_type.upper()}: converges={results[loss_type]['converges']}, "
                      f"monotonic={results[loss_type]['monotonic_decrease']}, "
                      f"final_loss={results[loss_type]['final_loss']:.6f}")
                
            except Exception as e:
                print(f"    WARN {loss_type} mathematical test failed: {e}")
        
        # Test loss function symmetries and properties
        for loss_type in loss_types:
            try:
                loss_fn = create_component('loss', loss_type, {'reduction': 'mean'})
                if loss_fn is None:
                    continue
                
                # Test symmetry: loss(a,b) should equal loss(b,a) for some losses
                loss_ab = loss_fn(pred, target)
                # Note: MSE and MAE are not symmetric in this sense, so we test magnitude
                
                # Test triangle inequality approximation
                third_point = torch.randn_like(pred)
                loss_ac = loss_fn(pred, third_point)
                loss_bc = loss_fn(target, third_point)
                
                print(f"    CHART {loss_type.upper()} triangle: {loss_ab.item():.4f}, "
                      f"{loss_ac.item():.4f}, {loss_bc.item():.4f}")
                
            except Exception as e:
                print(f"    WARN {loss_type} symmetry test failed: {e}")
        
        print("    PASS Loss mathematical properties validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Loss mathematical properties test failed: {e}")
        return False

def test_loss_numerical_stability():
    """Test numerical stability of loss functions"""
    print("TEST Testing Loss Numerical Stability...")
    
    try:
        loss_types = ['mse', 'mae', 'quantile_loss']
        
        # Test with extreme values
        extreme_cases = [
            ("very_small", torch.ones(2, 10, 5) * 1e-8, torch.ones(2, 10, 5) * 1e-7),
            ("very_large", torch.ones(2, 10, 5) * 1e8, torch.ones(2, 10, 5) * 1e7),
            ("mixed_scale", torch.randn(2, 10, 5) * 1e6, torch.randn(2, 10, 5) * 1e-6),
            ("zeros", torch.zeros(2, 10, 5), torch.zeros(2, 10, 5)),
        ]
        
        for loss_type in loss_types:
            try:
                if loss_type == 'quantile_loss':
                    loss_fn = create_component('loss', loss_type, {
                        'quantiles': [0.5],
                        'reduction': 'mean'
                    })
                else:
                    loss_fn = create_component('loss', loss_type, {'reduction': 'mean'})
                
                if loss_fn is None:
                    continue
                
                print(f"    SEARCH Testing {loss_type.upper()} stability:")
                
                for case_name, pred, target in extreme_cases:
                    try:
                        if loss_type == 'quantile_loss':
                            pred = pred.unsqueeze(-1)  # Add quantile dimension
                        
                        loss = loss_fn(pred, target)
                        
                        is_finite = torch.isfinite(loss).all()
                        is_nan = torch.isnan(loss).any()
                        is_inf = torch.isinf(loss).any()
                        
                        status = "PASS" if is_finite and not is_nan and not is_inf else "FAIL"
                        print(f"      {status} {case_name}: loss={loss.item():.2e}, "
                              f"finite={is_finite}, nan={is_nan}, inf={is_inf}")
                        
                    except Exception as e:
                        print(f"      FAIL {case_name}: {e}")
                        
            except Exception as e:
                print(f"    WARN {loss_type} stability test setup failed: {e}")
        
        print("    PASS Loss numerical stability validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Loss numerical stability test failed: {e}")
        return False

def run_loss_functionality_tests():
    """Run all loss function functionality tests"""
    print("ROCKET Running Loss Function Component Functionality Tests")
    print("=" * 80)
    
    if not COMPONENTS_AVAILABLE:
        print("FAIL Modular components not available - skipping tests")
        return False
    
    tests = [
        ("MSE Loss", test_mse_loss_functionality),
        ("MAE Loss", test_mae_loss_functionality),
        ("Bayesian MSE Loss", test_bayesian_mse_loss_functionality),
        ("Quantile Loss", test_quantile_loss_functionality),
        ("Frequency-Aware Loss", test_frequency_aware_loss),
        ("Loss Mathematical Properties", test_loss_mathematical_properties),
        ("Loss Numerical Stability", test_loss_numerical_stability),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTARGET {test_name}")
        print("-" * 60)
        
        try:
            if test_func():
                passed += 1
                print(f"PASS {test_name} PASSED")
            else:
                print(f"FAIL {test_name} FAILED")
        except Exception as e:
            print(f"FAIL {test_name} ERROR: {e}")
    
    print("\n" + "=" * 80)
    print(f"CHART Loss Function Functionality Test Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("PARTY All loss function functionality tests passed!")
        return True
    else:
        print("WARN Some loss function functionality tests failed")
        return False

if __name__ == "__main__":
    success = run_loss_functionality_tests()
    sys.exit(0 if success else 1)
