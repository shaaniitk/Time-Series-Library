"""
Output head registrations that expose utils output implementations to the unified registry.
"""
from __future__ import annotations

from utils.modular_components.registry import register_component
from utils.modular_components.implementations.outputs import (
    OutputConfig,
    ForecastingHead,
    RegressionHead,
    ClassificationHead,
    ProbabilisticForecastingHead,
)


def register_utils_outputs() -> None:
    """Register output heads into the global ComponentRegistry."""
    register_component(
        "output",
        "forecasting_head",
        ForecastingHead,
        metadata={
            "domain": "output",
            "task_types": ["forecasting"],
            "features": ["multistep", "activation_optional"],
            "source": "utils.modular_components.implementations.outputs",
        },
    )
    register_component(
        "output",
        "regression_head",
        RegressionHead,
        metadata={
            "domain": "output",
            "task_types": ["regression"],
            "features": ["sequence_or_single", "activation_optional"],
            "source": "utils.modular_components.implementations.outputs",
        },
    )
    register_component(
        "output",
        "classification_head",
        ClassificationHead,
        metadata={
            "domain": "output",
            "task_types": ["classification"],
            "features": ["sequence_or_pooled", "num_classes_param"],
            "source": "utils.modular_components.implementations.outputs",
        },
    )
    register_component(
        "output",
        "probabilistic_forecasting_head",
        ProbabilisticForecastingHead,
        metadata={
            "domain": "output",
            "task_types": ["probabilistic_forecasting"],
            "features": ["mean_logvar", "uncertainty"],
            "source": "utils.modular_components.implementations.outputs",
        },
    )
