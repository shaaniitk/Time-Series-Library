"""Decoder registry coverage tests (migrated slice).

Ensures registry enumerates components and basic forward shape integrity for
each available decoder using synthetic embeddings.
"""
from __future__ import annotations

import pytest
import torch
from typing import Dict, Any

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from layers.modular.decoder.registry import DecoderRegistry, get_decoder_component  # type: ignore
except Exception:  # pragma: no cover
    DecoderRegistry = None  # type: ignore
    get_decoder_component = None  # type: ignore


def test_decoder_registry_non_empty() -> None:
    if DecoderRegistry is None:
        pytest.skip("DecoderRegistry unavailable")
    names = DecoderRegistry.list_components()
    assert names, "No decoder components registered"


def _decoder_param_map() -> Dict[str, Any]:
    """Provide required args for decoder construction.

    Shared across all registered decoder variants (standard/enhanced/stable).
    """
    from layers.modular.attention import get_attention_component  # type: ignore
    from layers.modular.decomposition import get_decomposition_component  # type: ignore

    attention = get_attention_component("autocorrelation_layer", d_model=32, n_heads=2, dropout=0.0)
    # Use learnable decomposition so enhanced/stable decoders can query input_dim
    decomp = get_decomposition_component("learnable_decomp", input_dim=32, init_kernel_size=5, max_kernel_size=8)
    return {
        "d_layers": 1,
        "d_model": 32,
        "c_out": 32,
        "n_heads": 2,
        "d_ff": 64,
        "dropout": 0.0,
        "activation": "gelu",
        "self_attention_comp": attention,
        "cross_attention_comp": attention,
        "decomp_comp": decomp,
    }


@pytest.mark.parametrize("name", ["standard", "enhanced", "stable"])
def test_decoder_forward_minimal(name: str) -> None:
    if DecoderRegistry is None or get_decoder_component is None:
        pytest.skip("DecoderRegistry unavailable")
    if name not in DecoderRegistry.list_components():
        pytest.skip(f"Decoder {name} not present")

    dec = get_decoder_component(name, **_decoder_param_map())

    enc_out = torch.randn(2, 24, 32)
    tgt = torch.randn(2, 12, 32, requires_grad=True)
    # Provide zero trend to satisfy residual addition inside underlying Decoder
    init_trend = torch.zeros_like(tgt)
    raw_out = dec(tgt, enc_out, trend=init_trend)  # type: ignore[misc]

    # Decoders may return (seasonal, trend) or (seasonal, trend, aux)
    assert isinstance(raw_out, tuple) and len(raw_out) in (2, 3)
    seasonal, trend = raw_out[0], raw_out[1]
    assert seasonal.shape == tgt.shape
    assert trend.shape[0] == tgt.shape[0]
    assert torch.isfinite(seasonal).all() and torch.isfinite(trend).all()
    seasonal.sum().backward()
    assert tgt.grad is not None and torch.isfinite(tgt.grad).all()

