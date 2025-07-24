"""
Dynamic registry for all modular loss families
"""
from .bayesian_loss import BayesianMSELoss, BayesianMAELoss, BayesianLoss
from .quantile_loss import QuantileLoss, PinballLoss
from .basic_loss import MSELoss, MAELoss, HuberLoss, CrossEntropyLoss
from .advanced_loss import AdaptiveStructuralLoss, FrequencyAwareLoss
from .timeseries_loss import MAPELoss, SMAPELoss, MASELoss, DTWLoss
from .classification_loss import FocalLoss
from .specialized_loss import PSLoss, UncertaintyCalibrationLoss, AdaptiveAutoformerLoss, AdaptiveLoss

LOSS_REGISTRY = {
    # Basic Losses
    'mse': MSELoss,
    'mae': MAELoss,
    'huber': HuberLoss,
    'cross_entropy': CrossEntropyLoss,
    # Bayesian Losses
    'bayesian_mse': BayesianMSELoss,
    'bayesian_mae': BayesianMAELoss,
    'bayesian': BayesianLoss,
    # Quantile Losses
    'quantile': QuantileLoss,
    'pinball': PinballLoss,
    # Advanced Losses
    'adaptive_structural': AdaptiveStructuralLoss,
    'frequency_aware': FrequencyAwareLoss,
    # Time Series Losses
    'mape': MAPELoss,
    'smape': SMAPELoss,
    'mase': MASELoss,
    'dtw': DTWLoss,
    # Classification Losses
    'focal': FocalLoss,
    # Specialized Losses
    'ps_loss': PSLoss,
    'uncertainty_calibration': UncertaintyCalibrationLoss,
    'adaptive_autoformer': AdaptiveAutoformerLoss,
    'adaptive': AdaptiveLoss,
}

def get_loss_function(loss_type, config):
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return LOSS_REGISTRY[loss_type](config)

def list_available_losses():
    return list(LOSS_REGISTRY.keys())
