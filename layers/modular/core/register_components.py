"""Component registration for the unified registry.

This module registers all available components with the unified registry.
Import this module to populate the registry with all components.
"""

from .registry import component_registry, ComponentFamily
from ..fusion.hierarchical_fusion import HierarchicalFusion
from ..loss.quantile_loss import PinballLoss
from ..loss.standard_losses import StandardLossWrapper
from ..loss.advanced_losses import (
    MAPELoss, SMAPELoss, MASELoss, PSLoss, FocalLoss
)
from ..loss.adaptive_bayesian_losses import (
    AdaptiveAutoformerLoss, FrequencyAwareLoss, BayesianLoss,
    BayesianQuantileLoss, QuantileLoss, UncertaintyCalibrationLoss
)
import torch.nn as nn

# Register Fusion Components
component_registry.register(
    name="hierarchical_fusion",
    component_class=HierarchicalFusion,
    component_type=ComponentFamily.FUSION,
    test_config={
        "d_model": 512,
        "n_levels": 3,
        "fusion_strategy": "weighted_concat"
    }
)

# Register Loss Components
# Standard losses
component_registry.register(
    name="quantile_loss",
    component_class=PinballLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"quantiles": [0.1, 0.5, 0.9]}
)

component_registry.register(
    name="pinball_loss",  # Alias for quantile
    component_class=PinballLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"quantiles": [0.1, 0.5, 0.9]}
)

# Advanced metric losses
component_registry.register(
    name="mape_loss",
    component_class=MAPELoss,
    component_type=ComponentFamily.LOSS,
    test_config={}
)

component_registry.register(
    name="smape_loss",
    component_class=SMAPELoss,
    component_type=ComponentFamily.LOSS,
    test_config={}
)

component_registry.register(
    name="mase_loss",
    component_class=MASELoss,
    component_type=ComponentFamily.LOSS,
    test_config={"seasonal_periods": 1}
)

component_registry.register(
    name="ps_loss",
    component_class=PSLoss,
    component_type=ComponentFamily.LOSS,
    test_config={}
)

component_registry.register(
    name="focal_loss",
    component_class=FocalLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"alpha": 1.0, "gamma": 2.0}
)

# Adaptive losses
component_registry.register(
    name="adaptive_autoformer_loss",
    component_class=AdaptiveAutoformerLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"base_loss": "mse"}
)

component_registry.register(
    name="frequency_aware_loss",
    component_class=FrequencyAwareLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"base_loss": "mse"}
)

component_registry.register(
    name="multi_quantile_loss",
    component_class=QuantileLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"quantiles": [0.1, 0.5, 0.9]}
)

# Bayesian losses
component_registry.register(
    name="bayesian_quantile_loss",
    component_class=BayesianQuantileLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"quantiles": [0.1, 0.5, 0.9]}
)

component_registry.register(
    name="uncertainty_calibration_loss",
    component_class=UncertaintyCalibrationLoss,
    component_type=ComponentFamily.LOSS,
    test_config={"base_loss": "mse"}
)

# Additional fusion components can be registered here as they are added
# Example:
# component_registry.register(
#     name="attention_fusion",
#     component_class=AttentionFusion,
#     component_type=ComponentFamily.FUSION,
#     test_config={"d_model": 512, "n_heads": 8}
# )