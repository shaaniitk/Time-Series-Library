from __future__ import annotations

"""Encoder component tests.

Ensures minimal instantiation and forward for standard/enhanced/stable/hierarchical encoders
with synthetic dependencies.
"""

import pytest
import torch

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from layers.modular.encoder.registry import EncoderRegistry, get_encoder_component  # type: ignore
    from layers.modular.core import get_attention_component  # type: ignore
    import layers.modular.core.register_components  # noqa: F401  # populate registry side-effects
    from layers.modular.decomposition import get_decomposition_component  # type: ignore
except Exception:  # pragma: no cover
    EncoderRegistry = None  # type: ignore
    get_encoder_component = None  # type: ignore
    get_attention_component = None  # type: ignore
    get_decomposition_component = None  # type: ignore


def _has(name: str) -> bool:
    if EncoderRegistry is None:
        return False
    try:
        return name in EncoderRegistry.list_components()
    except Exception:
        return False


def _deps():
    attn = get_attention_component("autocorrelation_layer", d_model=32, n_heads=2, dropout=0.0)
    decomp = get_decomposition_component("series_decomp", kernel_size=3)
    return attn, decomp


@pytest.mark.parametrize("name", ["standard", "enhanced", "stable", "hierarchical"])
def test_encoder_minimal_forward(name: str) -> None:
    if EncoderRegistry is None or get_encoder_component is None or get_attention_component is None or get_decomposition_component is None:
        pytest.skip("EncoderRegistry unavailable")
    if not _has(name):
        pytest.skip(f"{name} encoder not registered")

    attn, decomp = _deps()
    if name == "hierarchical":
        kwargs = {
            "e_layers": 1,
            "d_model": 32,
            "n_heads": 2,
            "d_ff": 64,
            "dropout": 0.0,
            "activation": "gelu",
            "attention_type": "autocorrelation_layer",
            "decomp_type": "series_decomp",
            "decomp_params": {"kernel_size": 3},
            "n_levels": 2,
            "share_weights": False,
        }
    else:
        kwargs = {
            "e_layers": 1,
            "d_model": 32,
            "n_heads": 2,
            "d_ff": 64,
            "dropout": 0.0,
            "activation": "gelu",
            "attention_comp": attn,
            "decomp_comp": decomp,
        }

    enc = get_encoder_component(name, **kwargs)
    x = torch.randn(2, 16, 32, requires_grad=True)
    raw_out = enc(x)  # type: ignore[misc]
    out = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out
    assert out.shape == x.shape
    out.mean().backward()
    assert x.grad is not None
