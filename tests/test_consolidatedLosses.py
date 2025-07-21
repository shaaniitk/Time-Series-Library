import sys
import os
import pytest

# Ensure the parent directory is in sys.path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.modular_components.implementations import Losses

def test_loss_registry_keys():
    registry = Losses.LOSS_REGISTRY
    # Check for critical losses
    for key in ['bayesian_mse', 'bayesian_mae', 'mse', 'mae']:
        assert key in registry, f"Loss '{key}' missing from registry"


def test_loss_instantiation():
    config = Losses.LossConfig()
    # Try to instantiate each critical loss
    for key in ['bayesian_mse', 'bayesian_mae', 'mse', 'mae']:
        loss_fn = Losses.get_loss_function(key, config)
        assert hasattr(loss_fn, 'forward'), f"Loss '{key}' does not have a forward method"


def test_bayesian_mse_forward():
    import torch
    config = Losses.LossConfig()
    loss_fn = Losses.get_loss_function('bayesian_mse', config)
    pred = torch.randn(8, 10)
    true = torch.randn(8, 10)
    result = loss_fn.forward(pred, true)
    assert isinstance(result, torch.Tensor)


def test_loss_config_defaults():
    config = Losses.LossConfig()
    assert config.reduction == 'mean'
    assert config.ignore_index == -100

if __name__ == "__main__":
    pytest.main([__file__])
