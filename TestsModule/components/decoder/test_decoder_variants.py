from __future__ import annotations

"""Decoder component tests.

Basic forward shape checks for standard/enhanced/stable decoder variants.
"""

import pytest
import torch

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from layers.modular.decoder.registry import DecoderRegistry, get_decoder_component  # type: ignore
    from layers.modular.core import get_attention_component  # type: ignore
    import layers.modular.core.register_components  # noqa: F401  # populate registry side-effects
    from layers.modular.decomposition import get_decomposition_component  # type: ignore
except Exception:  # pragma: no cover
    DecoderRegistry = None  # type: ignore
    get_decoder_component = None  # type: ignore
    get_attention_component = None  # type: ignore
    get_decomposition_component = None  # type: ignore


def _has(name: str) -> bool:
    if DecoderRegistry is None:
        return False
    try:
        return name in DecoderRegistry.list_components()
    except Exception:
        return False


def _deps():
    attn = get_attention_component("autocorrelation_layer", d_model=32, n_heads=2, dropout=0.0)
    decomp = get_decomposition_component("learnable_decomp", input_dim=32, init_kernel_size=5, max_kernel_size=8)
    return attn, decomp


@pytest.mark.parametrize("name", ["standard", "enhanced", "stable"])
def test_decoder_minimal_forward(name: str) -> None:
    if DecoderRegistry is None or get_decoder_component is None or get_attention_component is None or get_decomposition_component is None:
        pytest.skip("DecoderRegistry unavailable")
    if not _has(name):
        pytest.skip(f"{name} decoder not registered")

    self_attn, decomp = _deps()
    kwargs = {
        "d_layers": 1,
        "d_model": 32,
        "c_out": 32,
        "n_heads": 2,
        "d_ff": 64,
        "dropout": 0.0,
        "activation": "gelu",
        "self_attention_comp": self_attn,
        "cross_attention_comp": self_attn,
        "decomp_comp": decomp,
    }

    dec = get_decoder_component(name, **kwargs)

    enc_out = torch.randn(2, 24, 32)
    tgt = torch.randn(2, 12, 32, requires_grad=True)
    init_trend = torch.zeros_like(tgt)
    raw_out = dec(tgt, enc_out, trend=init_trend)  # type: ignore[misc]
    seasonal, trend = raw_out[0], raw_out[1]
    assert seasonal.shape == tgt.shape
    assert trend.shape[0] == tgt.shape[0]
    seasonal.sum().backward()
    assert tgt.grad is not None
