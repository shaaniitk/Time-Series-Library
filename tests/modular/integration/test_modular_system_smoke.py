"""Modular System Smoke Tests

Validates that the component registry exposes expected core component types and
that basic create_component flows work with structured configs.
"""
from __future__ import annotations

import pytest

from configs.schemas import ComponentType, AttentionConfig, DecompositionConfig, LossConfig
from configs.modular_components import component_registry


def test_registry_basic_presence() -> None:
    """Ensure the registry is populated with a minimum expected subset."""
    assert component_registry._components, "Component registry should contain registered components"
    required = {ComponentType.AUTOCORRELATION, ComponentType.LEARNABLE_DECOMP, ComponentType.MSE}
    missing = required.difference(component_registry._components.keys())
    assert not missing, f"Missing required component types: {missing}"


@pytest.mark.parametrize(
    "attn_type,decomp_type,loss_type",
    [(ComponentType.AUTOCORRELATION, ComponentType.LEARNABLE_DECOMP, ComponentType.MSE)],
)
def test_create_core_components(attn_type: ComponentType, decomp_type: ComponentType, loss_type: ComponentType) -> None:
    """Instantiate a minimal trio of attention, decomposition, and loss components."""
    attn_cfg = AttentionConfig(type=attn_type, d_model=64, n_heads=4, dropout=0.1, factor=1, output_attention=False)
    attn = component_registry.create_component(attn_cfg.type, attn_cfg)
    assert attn is not None
    decomp_cfg = DecompositionConfig(type=decomp_type, kernel_size=7, input_dim=64)
    decomp = component_registry.create_component(decomp_cfg.type, decomp_cfg)
    assert decomp is not None
    loss_cfg = LossConfig(type=loss_type)
    loss_fn = component_registry.create_component(loss_cfg.type, loss_cfg)
    assert loss_fn is not None
