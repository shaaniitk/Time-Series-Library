"""Modular decomposition component smoke tests (migrated from root test_decomposition_components)."""
from __future__ import annotations

import torch
import pytest

from configs.modular_components import component_registry, register_all_components
from configs.schemas import ComponentType, DecompositionConfig

DECOMP_TYPES = [
    # Legacy SERIES_DECOMP / STABLE_DECOMP both map to the generic moving average implementation
    ComponentType.MOVING_AVG,
    ComponentType.LEARNABLE_DECOMP,
    ComponentType.WAVELET_DECOMP,
]

@pytest.mark.parametrize("decomp_type", DECOMP_TYPES)
def test_decomposition_component_forward(decomp_type: ComponentType) -> None:
    """Validate seasonal/trend decomposition outputs shapes & finiteness.

    Args:
        decomp_type: Enum specifying which decomposition to exercise.
    """
    register_all_components()

    batch_size, seq_len, d_model = 2, 96, 64
    x = torch.randn(batch_size, seq_len, d_model)

    if decomp_type == ComponentType.MOVING_AVG:
        config = DecompositionConfig(type=decomp_type, kernel_size=25)
        component = component_registry.create_component(decomp_type, config)
    elif decomp_type == ComponentType.LEARNABLE_DECOMP:
        # Provide input_dim via config so internal conv layers initialize with correct channels
        config = DecompositionConfig(type=decomp_type, kernel_size=25, input_dim=d_model)
        component = component_registry.create_component(decomp_type, config)
    else:  # WAVELET_DECOMP
        config = DecompositionConfig(type=decomp_type, kernel_size=25, levels=3)
        component = component_registry.create_component(decomp_type, config, d_model=d_model)

    with torch.no_grad():
        seasonal, trend = component(x)

    assert seasonal.shape == x.shape
    assert trend.shape == x.shape
    assert torch.isfinite(seasonal).all()
    assert torch.isfinite(trend).all()

    recon = seasonal + trend
    assert recon.shape == x.shape
