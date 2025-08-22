"""Encoder registry coverage tests.

Confirms encoder components can be enumerated and (where constructor
requirements allow) minimally instantiated. For encoders whose required
arguments are not trivially satisfiable (complex dependency graph), the test
xfails with a diagnostic instead of failing hard—surfacing API drift while
preserving suite stability during refactor.
"""
from __future__ import annotations

import pytest
import torch
from typing import Dict, Any

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from layers.modular.encoder.registry import EncoderRegistry, get_encoder_component  # type: ignore
except Exception:  # pragma: no cover
    EncoderRegistry = None  # type: ignore
    get_encoder_component = None  # type: ignore


def test_encoder_registry_non_empty() -> None:
    if EncoderRegistry is None:
        pytest.skip("EncoderRegistry unavailable")
    names = EncoderRegistry.list_components()
    assert names, "No encoder components registered"


def _encoder_param_map(name: str) -> Dict[str, Any]:
    """Return a minimal, valid constructor argument mapping per encoder.

    Ensures we provide required dependency objects (attention + decomposition)
    so tests assert real functionality instead of xfail placeholders.
    """
    from layers.modular.core import get_attention_component  # type: ignore
    import layers.modular.core.register_components  # noqa: F401  # populate registry side-effects
    from layers.modular.decomposition import get_decomposition_component  # type: ignore

    base = {
        "e_layers": 1,
        "d_model": 32,
        "n_heads": 2,
        "d_ff": 64,
        "dropout": 0.0,
        "activation": "gelu",
    }
    # Shared dependencies
    attention = get_attention_component("autocorrelation_layer", d_model=32, n_heads=2, dropout=0.0)
    decomp = get_decomposition_component("series_decomp", kernel_size=3)

    if name == "hierarchical":
        # Provide a lightweight Namespace mimicking configs consumed by enhancedcomponents.HierarchicalEncoder
        from argparse import Namespace
        # Also include direct modular signature args so either path succeeds
        return {
            "e_layers": 1,
            "d_model": 32,
            "n_heads": 2,
            "d_ff": 64,
            "dropout": 0.0,
            "activation": "gelu",
            "attention_type": "autocorrelation_layer",
            "decomp_type": "series_decomp",
            "decomp_params": {"kernel_size": 3},
            # Fallback enhancedcomponents signature bundle
            "configs": Namespace(
                e_layers=1,
                d_model=32,
                n_heads=2,
                d_ff=64,
                dropout=0.0,
                activation="gelu",
                factor=1,
                use_moe_ffn=False,
            ),
            "n_levels": 2,
            "share_weights": False,
        }
    # Standard / enhanced / stable use direct component instances
    return {**base, "attention_comp": attention, "decomp_comp": decomp}


@pytest.mark.parametrize("name", ["standard", "enhanced", "stable", "hierarchical"])
def test_encoder_minimal_instantiation_and_forward(name: str) -> None:
    if EncoderRegistry is None or get_encoder_component is None:
        pytest.skip("EncoderRegistry unavailable")
    if name not in EncoderRegistry.list_components():
        pytest.skip(f"Encoder {name} not registered")

    kwargs = _encoder_param_map(name)
    enc = get_encoder_component(name, **kwargs)

    x = torch.randn(2, 16, 32, requires_grad=True)
    raw_out = enc(x)  # type: ignore[misc]

    # Hierarchical encoder returns (tensor, None); others return tuple from underlying impl
    if isinstance(raw_out, tuple) or isinstance(raw_out, list):  # type: ignore[truthy-bool]
        out = raw_out[0]
    else:
        out = raw_out

    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
    out.mean().backward()
    assert x.grad is not None
