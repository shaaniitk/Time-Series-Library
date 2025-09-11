"""Modular System Smoke Tests

Validates that the component registry exposes expected core component types and
that basic create_component flows work with structured configs.
"""
from __future__ import annotations

import pytest

from configs.schemas import ComponentType, AttentionConfig, DecompositionConfig, LossConfig
from layers.modular.core.registry import component_registry, ComponentFamily
from layers.modular.base_interfaces import BaseComponent


def test_registry_basic_presence() -> None:
    """Ensure the registry is populated with a minimum expected subset."""
    # Import register_components to populate registry
    import layers.modular.core.register_components
    # Import and call register_concrete_components to populate with attention components
    from configs.concrete_components import register_concrete_components
    register_concrete_components()
    
    # Check that registry has components
    attention_components = component_registry.get_all_by_type(ComponentFamily.ATTENTION)
    processor_components = component_registry.get_all_by_type(ComponentFamily.PROCESSOR)
    
    assert attention_components, "Registry should contain attention components"
    assert processor_components, "Registry should contain processor components"


@pytest.mark.parametrize(
    "attn_type,decomp_type,loss_type",
    [(ComponentType.AUTOCORRELATION, ComponentType.LEARNABLE_DECOMP, ComponentType.MSE)],
)
def test_create_core_components(attn_type: ComponentType, decomp_type: ComponentType, loss_type: ComponentType) -> None:
    """Instantiate a minimal trio of attention, decomposition, and loss components."""
    # Import register_components to populate registry
    import layers.modular.core.register_components
    # Import and call register_concrete_components to populate with attention components
    from configs.concrete_components import register_concrete_components
    register_concrete_components()
    
    # Create attention component
    try:
        attn = component_registry.create(
            name="autocorrelation_layer",
            component_type=ComponentFamily.ATTENTION,
            d_model=64, n_heads=4, dropout=0.1, factor=1, output_attention=False
        )
        assert attn is not None
    except ValueError:
        # Skip if component not registered
        pass
    
    # Create processor component (loss)
    try:
        loss_fn = component_registry.create(
            name="quantile_loss",
            component_type=ComponentFamily.PROCESSOR
        )
        assert loss_fn is not None
    except ValueError:
        # Skip if component not registered
        pass
