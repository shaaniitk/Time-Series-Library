"""Tuning Component Registry

This module provides a unified registry for tuning components that optimize
model hyperparameters and training dynamics.
"""

from typing import Dict, Any, Type, Optional, List
from ..core.registry import ComponentFamily, register_component, get_component, list_components

# Import tuning implementations
try:
    from .kl_tuning import KLTuner
    TUNING_AVAILABLE = True
except ImportError as e:
    TUNING_AVAILABLE = False
    KLTuner = None
    print(f"Warning: Tuning components not available: {e}")


# Register tuning components
if TUNING_AVAILABLE:
    # KL Tuner
    register_component(
        ComponentFamily.TUNING,
        "kl_tuner",
        KLTuner,
        description="KL loss tuning utility for Bayesian models with adaptive weight adjustment",
        config_schema={
            "target_kl_percentage": {"type": "number", "default": 0.1, "description": "Target KL contribution percentage (0.1 = 10%)"},
            "min_weight": {"type": "number", "default": 1e-6, "description": "Minimum KL weight"},
            "max_weight": {"type": "number", "default": 1e-1, "description": "Maximum KL weight"}
        },
        tags=["kl-divergence", "bayesian", "regularization", "adaptive-tuning"]
    )


# Factory functions
def create_kl_tuner(model, config: Optional[Dict[str, Any]] = None) -> 'KLTuner':
    """Create a KL tuner with the given model and configuration."""
    if not TUNING_AVAILABLE or KLTuner is None:
        raise ImportError("KLTuner not available")
    
    config = config or {}
    target_kl_percentage = config.get('target_kl_percentage', 0.1)
    min_weight = config.get('min_weight', 1e-6)
    max_weight = config.get('max_weight', 1e-1)
    
    return KLTuner(
        model=model,
        target_kl_percentage=target_kl_percentage,
        min_weight=min_weight,
        max_weight=max_weight
    )


# Registry query functions
def list_tuning_components() -> List[str]:
    """List all registered tuning components."""
    return list_components(ComponentFamily.TUNING)


def get_tuning_component(name: str) -> Type:
    """Get a tuning component by name."""
    return get_component(ComponentFamily.TUNING, name)


def create_tuning_component(name: str, model, config: Optional[Dict[str, Any]] = None):
    """Create a tuning component by name with the given model and configuration."""
    if name == "kl_tuner":
        return create_kl_tuner(model, config)
    else:
        component_class = get_tuning_component(name)
        config = config or {}
        return component_class(model, **config)


# Export public interface
__all__ = [
    'create_kl_tuner',
    'list_tuning_components',
    'get_tuning_component',
    'create_tuning_component'
]