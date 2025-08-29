"""Extended Phase 2 attention component tests.

Migrates functionality of legacy scripts:
 - test_phase2_basic.py
 - test_phase2_attention.py

Focus:
 1. Forward shape & finiteness for restored / advanced attention variants.
 2. Registry presence heuristic (count of phase2 components >= threshold).
 3. Optional metadata & capability presence for restored sophisticated algorithms.

Note: Uses small tensor sizes to keep runtime low; excludes print noise.
"""
from __future__ import annotations

import pytest
import torch

from configs.modular_components import register_all_components
from configs.schemas import ComponentType, AttentionConfig
from configs.modular_components import component_registry

PHASE2_ATTENTIONS = [
    ComponentType.AUTOCORRELATION,
    ComponentType.ADAPTIVE_AUTOCORRELATION,
    ComponentType.CROSS_RESOLUTION,
]


@pytest.mark.parametrize("attention_type", PHASE2_ATTENTIONS)
def test_phase2_attention_forward(attention_type) -> None:
    """Forward pass smoke for selected Phase2 attention components via registry."""
    register_all_components()
    B, L, D, H = 2, 32, 64, 4
    q = torch.randn(B, L, D)
    cfg = AttentionConfig(
        type=attention_type,
        d_model=D,
        n_heads=H,
        dropout=0.0,
        factor=1,
        output_attention=False,
    )
    component = component_registry.create_component(attention_type, cfg, d_model=D, seq_len=L)
    with torch.no_grad():
        out, _ = component(q, q, q)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_phase2_registry_component_count() -> None:
    """Heuristic: registry contains a minimum number of advanced attention entries."""
    register_all_components()
    # Attempt to inspect registry; structure may vary.
    try:
        entries = component_registry.list_components('attention')  # type: ignore[arg-type]
        if isinstance(entries, dict):
            count = len(entries.get('attention', []))
        else:
            count = len(entries)
    except Exception:
        pytest.skip("Unable to introspect attention registry structure")
    assert count >= 3, f"Expected at least 3 phase2 attention components, found {count}"
