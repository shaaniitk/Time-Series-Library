"""Modular attention component smoke tests (migrated from root test_attention_components).

Covers:
- Autocorrelation
- Adaptive Autocorrelation
- Cross Resolution Attention

Ensures each registered component produces finite outputs with correct shape.
"""
from __future__ import annotations

import torch
import pytest

from configs.modular_components import component_registry, register_all_components
from configs.schemas import ComponentType, AttentionConfig

@pytest.mark.parametrize(
    "attention_type",
    [
        ComponentType.AUTOCORRELATION,
        ComponentType.ADAPTIVE_AUTOCORRELATION,
        ComponentType.CROSS_RESOLUTION,
    ],
)
def test_attention_component_forward(attention_type: ComponentType) -> None:
    """Basic forward/shape/value validation for a single attention component.

    Args:
        attention_type: The component enum member to validate.
    """
    register_all_components()

    batch_size, seq_len, d_model, n_heads = 2, 96, 64, 8
    x = torch.randn(batch_size, seq_len, d_model)
    x_cross = torch.randn(batch_size, seq_len, d_model)

    # Build config and create component via new registry API
    config = AttentionConfig(
        type=attention_type,
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.1,
        factor=1,
        output_attention=False,
    )
    component = component_registry.create_component(attention_type, config, d_model=d_model, seq_len=seq_len)

    with torch.no_grad():
        if attention_type == ComponentType.CROSS_RESOLUTION:
            out, weights = component(x, x_cross, x_cross)
        else:
            out, weights = component(x, x, x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    # weights may be None or tensor depending on implementation – just ensure no crash.

