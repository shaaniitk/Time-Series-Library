import torch
import pytest

from layers.modular.decoder.sequential_mixture_decoder import SequentialMixtureNLLLoss


def test_sequential_mdn_loss_decreases_with_better_means():
    """Loss should decrease when means are closer to targets."""
    B, T, num_targets, K = 3, 12, 4, 3
    torch.manual_seed(42)

    targets = torch.randn(B, T, num_targets)

    # Baseline random mixture params
    means_bad = torch.randn(B, T, num_targets, K)
    log_stds_bad = torch.randn(B, T, num_targets, K) * 0.5 - 1.0
    log_weights_bad = torch.randn(B, T, K)

    # Improved means: first component closer to targets
    means_good = means_bad.clone()
    means_good[..., 0] = targets
    log_stds_good = log_stds_bad.clone()
    log_weights_good = log_weights_bad.clone()
    log_weights_good[..., 0] = 0.0  # favor first component

    loss_fn = SequentialMixtureNLLLoss(reduction='mean')

    loss_bad = loss_fn((means_bad, log_stds_bad, log_weights_bad), targets)
    loss_good = loss_fn((means_good, log_stds_good, log_weights_good), targets)

    assert torch.isfinite(loss_bad) and torch.isfinite(loss_good)
    assert loss_good.item() < loss_bad.item(), "Improved means should reduce NLL"


def test_sequential_mdn_loss_none_reduction_shape():
    """Loss with reduction='none' should return [B, T] shape."""
    B, T, num_targets, K = 2, 8, 4, 2
    means = torch.randn(B, T, num_targets, K)
    log_stds = torch.randn(B, T, num_targets, K) * 0.3 - 1.2
    log_weights = torch.randn(B, T, K)
    targets = torch.randn(B, T, num_targets)

    loss_fn = SequentialMixtureNLLLoss(reduction='none')
    loss = loss_fn((means, log_stds, log_weights), targets)

    assert loss.shape == (B, T)
    assert torch.all(torch.isfinite(loss))