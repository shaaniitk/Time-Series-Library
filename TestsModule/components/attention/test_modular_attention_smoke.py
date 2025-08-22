"""Modular-only smoke tests for attention components.

Validates that attention implementations under layers/modular are registered
in the unified registry and support a tiny forward pass. Skips gracefully for
components not available in the current build.
"""
from __future__ import annotations

import pytest
import torch

# Ensure registry is populated
import layers.modular.core.register_components  # noqa: F401
from layers.modular.core import unified_registry, ComponentFamily
from layers.modular.attention.registry import get_attention_component  # modular helper for kwargs

pytestmark = [pytest.mark.smoke]

# Curated set of commonly present components; we dynamically filter to those actually registered.
CANDIDATES = [
    "fourier_attention",
    "wavelet_attention",
    "enhanced_autocorrelation",
    "bayesian_attention",
    "autocorrelation_layer",
    "adaptive_autocorrelation_layer",
    "cross_resolution_attention",
]


def _available(names: list[str]) -> list[str]:
    registered = set(unified_registry.list(ComponentFamily.ATTENTION)[ComponentFamily.ATTENTION.value])
    return [n for n in names if n in registered]


@pytest.mark.parametrize("name", _available(CANDIDATES))
def test_modular_attention_forward_smoke(name: str) -> None:
    # Minimal dims
    d_model, n_heads = 32, 2

    # Instantiate using modular helper (handles per-component kwargs shape)
    comp = get_attention_component(name, d_model=d_model, n_heads=n_heads)

    # Forward
    x = torch.randn(2, 8, d_model, requires_grad=True)
    out, weights = comp(x, x, x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # Weights may be None; if tensor, ensure finite
    if isinstance(weights, torch.Tensor):
        assert torch.isfinite(weights).all()
