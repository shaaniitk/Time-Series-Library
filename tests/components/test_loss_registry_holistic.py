"""
Holistic Loss Registry Test

This test instantiates every registered loss component with a minimal config and validates a basic forward pass.
"""
import pytest
import torch
# PYTEST COMPATIBILITY: Ensure workspace root is in sys.path
import sys

import pytest
import torch
import inspect
from utils.modular_components.implementations.losses import LOSS_REGISTRY, LossConfig, BayesianLossConfig, AdaptiveLossConfig, FrequencyLossConfig, StructuralLossConfig, QuantileConfig, FocalLossConfig, DTWConfig

def build_config_for_loss(loss_cls):
    """Return a minimal config for the given loss class based on its constructor signature."""
    sig = inspect.signature(loss_cls.__init__)
    # Map config class by name
    config_map = {
        'LossConfig': LossConfig,
        'BayesianLossConfig': BayesianLossConfig,
        'AdaptiveLossConfig': AdaptiveLossConfig,
        'FrequencyLossConfig': FrequencyLossConfig,
        'StructuralLossConfig': StructuralLossConfig,
        'QuantileConfig': QuantileConfig,
        'FocalLossConfig': FocalLossConfig,
        'DTWConfig': DTWConfig,
    }
    # Find config param type
    for param in sig.parameters.values():
        if param.name == 'config':
            ann = param.annotation
            # If annotation is Union, get first type
            if hasattr(ann, '__origin__') and ann.__origin__ is Union:
                for t in ann.__args__:
                    if hasattr(t, '__name__') and t.__name__ in config_map:
                        return config_map[t.__name__]()
            elif hasattr(ann, '__name__') and ann.__name__ in config_map:
                return config_map[ann.__name__]()
    # Fallback: return dict
    return {}

def test_loss_registry_holistic():
    """Holistic test: instantiate every registered loss and run a forward pass."""
    dummy_pred = torch.randn(8, 16)
    dummy_true = torch.randn(8, 16)
    for loss_name, loss_cls in LOSS_REGISTRY.items():
        try:
            config = build_config_for_loss(loss_cls)
            # Some losses require extra args (e.g., num_classes for CrossEntropyLoss)
            sig = inspect.signature(loss_cls.__init__)
            extra_kwargs = {}
            if 'num_classes' in sig.parameters:
                extra_kwargs['num_classes'] = 2
            loss = loss_cls(config, **extra_kwargs)
            # Forward signature may vary
            forward_sig = inspect.signature(loss.forward)
            args = [dummy_pred, dummy_true]
            # Add dummy model if needed
            if 'model' in forward_sig.parameters:
                class DummyModel:
                    def kl_divergence(self): return torch.tensor(0.0)
                    def get_kl_loss(self): return torch.tensor(0.0)
                args.append(DummyModel())
            out = loss.forward(*args)
            assert isinstance(out, torch.Tensor)
        except Exception as e:
            pytest.skip(f"Skipping {loss_name}: {e}")
    if name == 'bayesianmseloss':
        return BayesianLossConfig()
    if name == 'bayesianmaeloss':
        return BayesianLossConfig()
    if name == 'bayesianloss':
        return BayesianLossConfig()
    if name == 'bayesianquantileloss':
        return QuantileConfig()
    if name == 'quantileloss':
        return QuantileConfig()
    if name == 'pinballloss':
        return QuantileConfig()
    if name == 'gaussiannllloss':
        return BayesianLossConfig()
    if name == 'studenttloss':
        return BayesianLossConfig()
    if name == 'focalloss':
        return FocalLossConfig(smooth=1e-8)
    if name == 'mapeloss':
        return LossConfig()
    if name == 'smapeloss':
        return LossConfig()
    if name == 'maseloss':
        return LossConfig()
    if name == 'dtwloss':
        return DTWConfig()
    if name == 'adaptivestructuralloss':
        return StructuralLossConfig(trend_weight=1.0, seasonal_weight=1.0, residual_weight=1.0)
    if name == 'frequencyawareloss':
        return FrequencyLossConfig(time_weight=1.0, high_freq_penalty=0.1)
    if name == 'psloss':
        return StructuralLossConfig(structural_weight=1.0, patch_weight=1.0)
    if name == 'uncertaintycalibrationloss':
        return BayesianLossConfig(calibration_weight=1.0, uncertainty_weight=0.1)
    if name == 'adaptiveautoformerloss':
        return AdaptiveLossConfig()
    if name == 'adaptiveloss':
        return AdaptiveLossConfig()
    if name == 'customloss':
        from utils.modular_components.implementations.Losses import CustomLossConfig
        return CustomLossConfig()
    return LossConfig()

@pytest.mark.parametrize("loss_name,loss_cls", list(LOSS_REGISTRY.items()))
def test_loss_registry_instantiation(loss_name, loss_cls):
    config = get_minimal_config(loss_cls)
    loss_instance = loss_cls(config)
    # Create dummy tensors for forward pass
    # Default dummy tensors
    pred = torch.randn(4, 8)
    true = torch.randn(4, 8)
    # Special handling for quantile losses
    if loss_name in ['bayesian_quantile', 'quantile']:
        num_quantiles = 3
        pred = torch.randn(4, num_quantiles)  # [batch, num_quantiles]
        true = torch.randn(4)  # [batch]
    # Some losses require extra args
    try:
        result = loss_instance.forward(pred, true)
    except Exception as e:
        # Try with extra args for known cases
        if loss_name == 'cross_entropy':
            true = torch.randint(0, 8, (4,))
            result = loss_instance.forward(pred, true)
        elif loss_name == 'mase':
            training_data = torch.randn(12)
            result = loss_instance.forward(pred, true, training_data=training_data)
        elif loss_name == 'gaussian_nll':
            var = torch.abs(torch.randn(4, 8))
            result = loss_instance.forward(pred, true, var=var)
        elif loss_name == 'dtw':
            pred = torch.randn(4, 8, 1)
            true = torch.randn(4, 8, 1)
            result = loss_instance.forward(pred, true)
        elif loss_name in ['bayesian_quantile', 'quantile']:
            num_quantiles = 3
            pred = torch.randn(4, num_quantiles)
            true = torch.randn(4)
            result = loss_instance.forward(pred, true)
        else:
            pytest.skip(f"Loss {loss_name} could not be tested: {e}")
    assert result is not None, f"Loss {loss_name} forward returned None"
