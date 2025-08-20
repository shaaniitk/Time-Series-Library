# Backward-compatible shim for Bayesian loss imports
# Some modules expect `layers.modular.losses.bayesian_loss`.
# The actual implementations live in `adaptive_bayesian_losses.py`.

from .adaptive_bayesian_losses import (
    BayesianLoss,
    BayesianQuantileLoss,
    KLTuner,
)

__all__ = [
    "BayesianLoss",
    "BayesianQuantileLoss",
    "KLTuner",
]