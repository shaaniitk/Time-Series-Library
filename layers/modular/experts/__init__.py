"""
Mixture of Experts (MoE) Framework for Time Series Models

This module provides a comprehensive framework for implementing Mixture of Experts
in time series forecasting models, with specialized experts for different aspects
of temporal and spatial modeling.
"""

from .base_expert import BaseExpert, ExpertOutput
from .expert_router import ExpertRouter, AdaptiveExpertRouter, AttentionBasedRouter
from .moe_layer import MoELayer, SparseMoELayer
from .registry import ExpertRegistry

# Temporal Experts
from .temporal.seasonal_expert import SeasonalPatternExpert
from .temporal.trend_expert import TrendPatternExpert
from .temporal.volatility_expert import VolatilityPatternExpert
from .temporal.regime_expert import RegimePatternExpert

# Spatial Experts
from .spatial.local_expert import LocalSpatialExpert
from .spatial.global_expert import GlobalSpatialExpert
from .spatial.hierarchical_expert import HierarchicalSpatialExpert

# Uncertainty Experts
from .uncertainty.aleatoric_expert import AleatoricUncertaintyExpert
from .uncertainty.epistemic_expert import EpistemicUncertaintyExpert

__all__ = [
    'BaseExpert', 'ExpertOutput',
    'ExpertRouter', 'AdaptiveExpertRouter', 'AttentionBasedRouter',
    'MoELayer', 'SparseMoELayer',
    'ExpertRegistry',
    'SeasonalPatternExpert', 'TrendPatternExpert', 'VolatilityPatternExpert', 'RegimePatternExpert',
    'LocalSpatialExpert', 'GlobalSpatialExpert', 'HierarchicalSpatialExpert',
    'AleatoricUncertaintyExpert', 'EpistemicUncertaintyExpert'
]