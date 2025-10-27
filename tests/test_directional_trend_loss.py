"""
Unit tests for DirectionalTrendLoss and HybridMDNDirectionalLoss.

Tests each component separately and validates integration with MDN outputs.
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from layers.modular.losses.directional_trend_loss import (
    DirectionalTrendLoss,
    HybridMDNDirectionalLoss,
    compute_directional_accuracy
)


class TestDirectionalAccuracyComponent:
    """Test the directional accuracy computation."""
    
    def test_perfect_direction_match(self):
        """Test that perfectly matching directions give low loss."""
        loss_fn = DirectionalTrendLoss(
            direction_weight=1.0,
            trend_weight=0.0,
            magnitude_weight=0.0
        )
        
        # Create predictions and targets with same directional pattern
        # Both increase: [1, 2, 3, 4, 5]
        pred = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).transpose(1, 2)  # [1, 5, 1]
        target = torch.tensor([[[1.0, 2.5, 3.5, 4.5, 5.5]]]).transpose(1, 2)  # [1, 5, 1]
        
        loss = loss_fn(pred, target)
        
        # Loss should be negative (because -tanh(+) * tanh(+) = -1)
        assert loss.item() < 0, "Perfect directional match should have negative loss"
    
    def test_opposite_direction_penalty(self):
        """Test that opposite directions give high loss."""
        loss_fn = DirectionalTrendLoss(
            direction_weight=1.0,
            trend_weight=0.0,
            magnitude_weight=0.0
        )
        
        # Prediction increases, target decreases
        pred = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).transpose(1, 2)  # [1, 5, 1]
        target = torch.tensor([[[5.0, 4.0, 3.0, 2.0, 1.0]]]).transpose(1, 2)  # [1, 5, 1]
        
        loss = loss_fn(pred, target)
        
        # Loss should be positive (because -tanh(+) * tanh(-) = +1)
        assert loss.item() > 0, "Opposite directions should have positive loss"
    
    def test_directional_loss_differentiable(self):
        """Test that directional loss has non-zero gradients."""
        loss_fn = DirectionalTrendLoss(
            direction_weight=1.0,
            trend_weight=0.0,
            magnitude_weight=0.0
        )
        
        pred = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).transpose(1, 2)  # [1, 5, 1]
        pred.requires_grad = True
        target = torch.tensor([[[1.0, 2.5, 3.5, 4.5, 5.5]]]).transpose(1, 2)  # [1, 5, 1]
        
        loss = loss_fn(pred, target)
        loss.backward()
        
        assert pred.grad is not None, "Gradients should be computed"
        assert torch.any(pred.grad != 0), "Gradients should be non-zero"


class TestTrendCorrelationComponent:
    """Test the trend correlation computation."""
    
    def test_perfect_correlation(self):
        """Test that perfectly correlated trends give low loss."""
        loss_fn = DirectionalTrendLoss(
            direction_weight=0.0,
            trend_weight=1.0,
            magnitude_weight=0.0,
            correlation_type='pearson'
        )
        
        # Non-constant differences with same pattern (perfect correlation)
        # Differences: [1, 2, 3, 4] for both (scaled)
        pred = torch.tensor([[[1.0, 2.0, 4.0, 7.0, 11.0]]]).transpose(1, 2)  # [1, 5, 1] - diffs: 1, 2, 3, 4
        target = torch.tensor([[[2.0, 4.0, 8.0, 14.0, 22.0]]]).transpose(1, 2)  # [1, 5, 1] - diffs: 2, 4, 6, 8 (2x scaled)
        
        loss = loss_fn(pred, target)
        
        # Loss should be close to 0 (1 - correlation where correlation ≈ 1)
        assert loss.item() < 0.1, f"Perfect correlation should give low loss, got {loss.item()}"
    
    def test_negative_correlation(self):
        """Test that negatively correlated trends give high loss."""
        loss_fn = DirectionalTrendLoss(
            direction_weight=0.0,
            trend_weight=1.0,
            magnitude_weight=0.0,
            correlation_type='pearson'
        )
        
        # Truly opposite patterns:
        # pred diffs: [1, 3, 6, 10] (accelerating)
        # target diffs: [1, 2, 3, 4] (constant acceleration)
        # These have different correlation patterns
        pred = torch.tensor([[[1.0, 2.0, 5.0, 11.0, 21.0]]]).transpose(1, 2)  # diffs: 1, 3, 6, 10
        target = torch.tensor([[[1.0, 2.0, 4.0, 7.0, 11.0]]]).transpose(1, 2)  # diffs: 1, 2, 3, 4
        
        loss_accelerating = loss_fn(pred, target)
        
        # Now test with truly opposite signs (one increasing, one decreasing)
        pred2 = torch.tensor([[[1.0, 3.0, 6.0, 10.0, 15.0]]]).transpose(1, 2)  # diffs: 2, 3, 4, 5
        target2 = torch.tensor([[[15.0, 13.0, 10.0, 6.0, 1.0]]]).transpose(1, 2)  # diffs: -2, -3, -4, -5
        
        loss_opposite = loss_fn(pred2, target2)
        
        # Opposite signs should have higher loss than different patterns with same sign
        assert loss_opposite > loss_accelerating, f"Opposite signs should give higher loss: {loss_opposite} vs {loss_accelerating}"
        # Loss for opposite signs should be substantial
        assert loss_opposite > 1.0, f"Opposite sign correlation should give loss > 1.0, got {loss_opposite}"
    
    def test_spearman_correlation(self):
        """Test that Spearman correlation works."""
        loss_fn = DirectionalTrendLoss(
            direction_weight=0.0,
            trend_weight=1.0,
            magnitude_weight=0.0,
            correlation_type='spearman'
        )
        
        # Non-linear but monotonic increase
        pred = torch.tensor([[[1.0, 4.0, 9.0, 16.0, 25.0]]]).transpose(1, 2)  # [1, 5, 1] - squared
        target = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).transpose(1, 2)  # [1, 5, 1] - linear
        
        loss = loss_fn(pred, target)
        
        # Spearman should handle non-linear monotonic relationships well
        assert loss.item() < 0.2, f"Spearman should handle monotonic relationships, got {loss.item()}"


class TestMagnitudeComponent:
    """Test the magnitude loss computation."""
    
    def test_exact_match(self):
        """Test that exact matches give zero magnitude loss."""
        loss_fn = DirectionalTrendLoss(
            direction_weight=0.0,
            trend_weight=0.0,
            magnitude_weight=1.0
        )
        
        pred = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).transpose(1, 2)  # [1, 5, 1]
        target = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).transpose(1, 2)  # [1, 5, 1]
        
        loss = loss_fn(pred, target)
        
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), "Exact match should give zero magnitude loss"
    
    def test_magnitude_scales_with_error(self):
        """Test that magnitude loss scales with prediction error."""
        loss_fn = DirectionalTrendLoss(
            direction_weight=0.0,
            trend_weight=0.0,
            magnitude_weight=1.0
        )
        
        pred = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).transpose(1, 2)  # [1, 5, 1]
        target = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).transpose(1, 2)  # [1, 5, 1]
        
        # Small error
        loss_small = loss_fn(pred, target + 0.1)
        
        # Large error
        loss_large = loss_fn(pred, target + 1.0)
        
        assert loss_large > loss_small, "Larger errors should give larger loss"


class TestMDNIntegration:
    """Test integration with MDN outputs."""
    
    def test_mdn_mean_extraction(self):
        """Test that MDN mean is correctly extracted."""
        loss_fn = DirectionalTrendLoss(
            direction_weight=1.0,
            trend_weight=0.0,
            magnitude_weight=0.0,
            use_mdn_mean=True
        )
        
        batch_size, seq_len, num_components = 2, 10, 3
        
        # Create MDN parameters
        means = torch.randn(batch_size, seq_len, num_components)
        log_stds = torch.randn(batch_size, seq_len, num_components)
        log_weights = torch.randn(batch_size, seq_len, num_components)
        
        target = torch.randn(batch_size, seq_len)
        
        # Should not raise error
        loss = loss_fn((means, log_stds, log_weights), target.unsqueeze(-1))
        
        assert loss is not None, "Loss should be computed from MDN parameters"
        assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_multivariate_mdn(self):
        """Test MDN with multiple target features."""
        loss_fn = DirectionalTrendLoss(
            direction_weight=1.0,
            trend_weight=0.0,
            magnitude_weight=0.0,
            use_mdn_mean=True
        )
        
        batch_size, seq_len, num_targets, num_components = 2, 10, 4, 3
        
        # Multivariate MDN parameters
        means = torch.randn(batch_size, seq_len, num_targets, num_components)
        log_stds = torch.randn(batch_size, seq_len, num_targets, num_components)
        log_weights = torch.randn(batch_size, seq_len, num_components)  # Shared across targets
        
        target = torch.randn(batch_size, seq_len, num_targets)
        
        # Should not raise error
        loss = loss_fn((means, log_stds, log_weights), target)
        
        assert loss is not None, "Loss should be computed from multivariate MDN parameters"
        assert not torch.isnan(loss), "Loss should not be NaN"


class TestHybridLoss:
    """Test the hybrid MDN + directional loss."""
    
    def test_hybrid_combination(self):
        """Test that hybrid loss combines both components."""
        hybrid_loss = HybridMDNDirectionalLoss(
            nll_weight=0.5,
            direction_weight=2.0,
            trend_weight=1.0,
            magnitude_weight=0.1
        )
        
        batch_size, seq_len, num_components = 2, 10, 3
        
        # Create MDN parameters with requires_grad
        means = torch.randn(batch_size, seq_len, num_components, requires_grad=True)
        log_stds = torch.rand(batch_size, seq_len, num_components, requires_grad=True) - 2  # Negative for small stds
        log_weights = torch.randn(batch_size, seq_len, num_components, requires_grad=True)
        
        target = torch.randn(batch_size, seq_len)
        
        loss = hybrid_loss((means, log_stds, log_weights), target.unsqueeze(-1))
        
        assert loss is not None, "Hybrid loss should be computed"
        assert not torch.isnan(loss), "Hybrid loss should not be NaN"
        
        # Test gradients by backpropagation
        loss.backward()
        assert means.grad is not None, "Hybrid loss should support gradients"
    
    def test_hybrid_with_ohlc(self):
        """Test hybrid loss with OHLC (4 target) data."""
        hybrid_loss = HybridMDNDirectionalLoss(
            nll_weight=0.3,
            direction_weight=3.0,
            trend_weight=1.5,
            magnitude_weight=0.1
        )
        
        batch_size, seq_len, num_targets, num_components = 4, 96, 4, 3
        
        # OHLC data simulation
        means = torch.randn(batch_size, seq_len, num_targets, num_components)
        log_stds = torch.rand(batch_size, seq_len, num_targets, num_components) - 2
        log_weights = torch.randn(batch_size, seq_len, num_components)
        
        target = torch.randn(batch_size, seq_len, num_targets)
        
        loss = hybrid_loss((means, log_stds, log_weights), target)
        
        assert loss is not None, "Hybrid loss should work with OHLC data"
        assert not torch.isnan(loss), "Hybrid loss should not be NaN for OHLC"


class TestDirectionalAccuracyMetric:
    """Test the directional accuracy metric utility."""
    
    def test_perfect_accuracy(self):
        """Test 100% directional accuracy."""
        # Both increase monotonically
        pred = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).transpose(1, 2)
        target = torch.tensor([[[1.0, 2.5, 3.5, 4.5, 5.5]]]).transpose(1, 2)
        
        accuracy = compute_directional_accuracy(pred, target)
        
        assert accuracy == 100.0, f"Perfect direction match should give 100% accuracy, got {accuracy}"
    
    def test_zero_accuracy(self):
        """Test 0% directional accuracy."""
        # Opposite directions
        pred = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).transpose(1, 2)
        target = torch.tensor([[[5.0, 4.0, 3.0, 2.0, 1.0]]]).transpose(1, 2)
        
        accuracy = compute_directional_accuracy(pred, target)
        
        assert accuracy == 0.0, f"Opposite directions should give 0% accuracy, got {accuracy}"
    
    def test_multivariate_accuracy(self):
        """Test directional accuracy with multiple targets."""
        batch_size, seq_len, num_targets = 4, 96, 4
        
        # Random predictions and targets
        pred = torch.randn(batch_size, seq_len, num_targets)
        target = torch.randn(batch_size, seq_len, num_targets)
        
        accuracy = compute_directional_accuracy(pred, target)
        
        # Accuracy should be between 0 and 100
        assert 0 <= accuracy <= 100, f"Accuracy should be in [0, 100], got {accuracy}"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_changes(self):
        """Test handling of zero changes (flat prediction/target)."""
        loss_fn = DirectionalTrendLoss(
            direction_weight=1.0,
            trend_weight=1.0,
            magnitude_weight=1.0
        )
        
        # Flat prediction and target
        pred = torch.ones(2, 10, 1)
        target = torch.ones(2, 10, 1)
        
        # Should not raise error or produce NaN
        loss = loss_fn(pred, target)
        
        assert not torch.isnan(loss), "Loss should not be NaN for flat inputs"
    
    def test_gradient_flow(self):
        """Test that gradients flow through all components."""
        loss_fn = DirectionalTrendLoss(
            direction_weight=5.0,
            trend_weight=2.0,
            magnitude_weight=0.1
        )
        
        pred = torch.randn(4, 96, 4, requires_grad=True)
        target = torch.randn(4, 96, 4)
        
        loss = loss_fn(pred, target)
        loss.backward()
        
        assert pred.grad is not None, "Gradients should be computed"
        assert torch.all(torch.isfinite(pred.grad)), "Gradients should be finite"
    
    def test_per_target_weights(self):
        """Test per-target weight functionality."""
        # Close price (index 3) weighted more heavily
        per_target_weights = torch.tensor([0.5, 0.5, 0.5, 1.0])
        
        loss_fn = DirectionalTrendLoss(
            direction_weight=1.0,
            trend_weight=1.0,
            magnitude_weight=1.0,
            per_target_weights=per_target_weights
        )
        
        pred = torch.randn(2, 10, 4)
        target = torch.randn(2, 10, 4)
        
        loss = loss_fn(pred, target)
        
        assert loss is not None, "Loss should be computed with per-target weights"
        assert not torch.isnan(loss), "Loss should not be NaN with per-target weights"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
