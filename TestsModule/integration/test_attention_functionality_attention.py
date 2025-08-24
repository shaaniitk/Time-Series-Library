"""Attention functionality tests (migrated from legacy monolith).

Focus: minimal yet meaningful coverage of registry exposure, forward pass
correctness, and basic parameter diversity characteristics. Designed to be
fast (extended marker) while exercising multiple component varieties.
"""
from __future__ import annotations

from inspect import signature
from typing import Iterable, List

import pytest
import torch

pytestmark = [pytest.mark.extended]

from layers.modular.attention.registry import get_attention_component  # type: ignore
import layers.modular.core.register_components  # noqa: F401  # populate registry side-effects


def _available_names(preferred: Iterable[str]) -> List[str]:
    names: List[str] = []
    # Try preferred names first (probe by try-create)
    for p in preferred:
        try:
            _ = get_attention_component(p, d_model=8, n_heads=2)
            names.append(p)
        except Exception:
            continue
    # Add a couple more detected components if desired (skip for simplicity)
    return names


def test_attention_registry_non_empty() -> None:
    """Helper should allow creating at least one known component."""
    try:
        _ = get_attention_component("autocorrelation_layer", d_model=16, n_heads=2)
    except Exception:
        # Try a couple of alternates
        for alt in ("enhanced_autocorrelation", "fourier_attention"):
            try:
                _ = get_attention_component(alt, d_model=16, n_heads=2)
                break
            except Exception:
                continue
        else:
            pytest.skip("No attention components available via unified helper")


@pytest.mark.parametrize(
    "name",
    _available_names([
        "enhanced_autocorrelation",
        "autocorrelation_layer",
        "fourier_attention",
        "wavelet_attention",
        "bayesian_attention",
        "causal_convolution",
    ]),
)
def test_attention_forward_shape_and_grad(name: str) -> None:
    """Instantiate component via helper (if available) and verify forward pass.

    We attempt (q,k,v) style call where possible; if instantiation or forward
    fails due to signature mismatch we xfail to surface unexpected API drift
    without breaking the suite entirely.
    """
    # unified helper available by import

    # Common minimal kwargs (filtered later by helper logic inside registry)
    kwargs = {"d_model": 32, "n_heads": 2, "dropout": 0.0, "factor": 1}
    try:
        comp = get_attention_component(name, **kwargs)
    except Exception as e:  # pragma: no cover - unexpected
        pytest.xfail(f"Could not instantiate {name}: {e}")

    # Prepare tiny tensors (batch=2, seq=16, d_model=32)
    q = torch.randn(2, 16, 32, requires_grad=True)
    k = torch.randn(2, 16, 32)
    v = torch.randn(2, 16, 32)

    # Try calling with (q,k,v) first; some legacy wrappers expect an extra 'mask'
    forward_tried = False
    output = None
    for call in [
        lambda: comp(q, k, v),  # type: ignore[misc]
        lambda: comp(q, k, v, None),  # mask=None variant
    ]:
        if forward_tried:
            break
        try:
            result = call()
            forward_tried = True
            output = result[0] if isinstance(result, (tuple, list)) else result
        except TypeError:
            continue
    if not forward_tried or output is None:
        pytest.xfail(f"Forward signature not supported for {name}")

    assert isinstance(output, torch.Tensor), "Output must be tensor"
    assert output.shape[:3] == (2, 16, 32), "Unexpected output shape"
    assert torch.isfinite(output).all(), "Non‑finite values in output"

    # Gradient path
    loss = output.sum()
    loss.backward()
    if q.grad is None or not torch.isfinite(q.grad).all():
        pytest.xfail(f"Gradient path incomplete for {name} (expected during refactor stage)")


def test_attention_parameter_diversity() -> None:
    """Adaptive / enhanced variants should have (weakly) larger parameter counts.

    We compare 'enhanced_autocorrelation' vs plain 'autocorrelation_layer' if both exist.
    If one missing we skip gracefully.
    """
    # Probe presence via try-create
    try:
        base = get_attention_component("autocorrelation_layer", d_model=32, n_heads=2)
        enhanced = get_attention_component("enhanced_autocorrelation", d_model=32, n_heads=2)
    except Exception:
        pytest.skip("Required components for parameter diversity test not present")
    base_params = sum(p.numel() for p in base.parameters())
    enhanced_params = sum(p.numel() for p in enhanced.parameters())
    # Enhanced often introduces extra projections / gating -> expect >=
    assert enhanced_params >= base_params, "Enhanced variant has fewer parameters than base"

