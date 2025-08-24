"""
Loss wrappers/registrations that expose loss implementations to the unified utils registry.

This registers standard, robust, probabilistic, and quantile losses under the 'loss' type.
"""
from __future__ import annotations

from typing import Dict, Any

from layers.modular.core.registry import register_component
from layers.modular.loss import (
    LossConfig,
    MSELoss,
    MAELoss,
    CrossEntropyLoss,
    HuberLoss,
    NegativeLogLikelihood,
    QuantileLoss,
    MultiTaskLoss,
)


def register_utils_losses() -> None:
    """Register loss implementations into the global ComponentRegistry."""
    # Regression losses
    register_component(
        "loss",
        "mse_loss",
        MSELoss,
        metadata={
            "domain": "loss",
            "task_types": ["regression", "forecasting"],
            "features": ["differentiable", "stable"],
            "source": "layers.modular.losses.modular_standard_losses",
        },
    )
    register_component(
        "loss",
        "mae_loss",
        MAELoss,
        metadata={
            "domain": "loss",
            "task_types": ["regression", "forecasting"],
            "features": ["robust_to_outliers", "differentiable"],
            "source": "layers.modular.losses.modular_standard_losses",
        },
    )
    register_component(
        "loss",
        "huber_loss",
        HuberLoss,
        metadata={
            "domain": "loss",
            "task_types": ["regression", "forecasting"],
            "features": ["robust_to_outliers", "delta_param"],
            "source": "layers.modular.losses.modular_standard_losses",
        },
    )

    # Classification
    register_component(
        "loss",
        "cross_entropy_loss",
        CrossEntropyLoss,
        metadata={
            "domain": "loss",
            "task_types": ["classification"],
            "features": ["class_weights", "ignore_index"],
            "source": "layers.modular.losses.modular_standard_losses",
        },
    )

    # Probabilistic
    register_component(
        "loss",
        "negative_log_likelihood_loss",
        NegativeLogLikelihood,
        metadata={
            "domain": "loss",
            "task_types": ["probabilistic_forecasting", "probabilistic_regression"],
            "features": ["requires_uncertainty", "gaussian"],
            "source": "layers.modular.losses.modular_standard_losses",
        },
    )

    # Quantile
    register_component(
        "loss",
        "quantile_loss",
        QuantileLoss,
        metadata={
            "domain": "loss",
            "task_types": ["quantile_regression", "probabilistic_forecasting"],
            "features": ["provides_uncertainty", "multi_quantile"],
            "source": "layers.modular.losses.modular_standard_losses",
        },
    )

    # Multi-task
    register_component(
        "loss",
        "multitask_loss",
        MultiTaskLoss,
        metadata={
            "domain": "loss",
            "task_types": ["multi_task"],
            "features": ["weighted_sum", "composite"],
            "source": "layers.modular.losses.modular_standard_losses",
        },
    )
