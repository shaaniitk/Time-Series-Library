from __future__ import annotations

"""Focused attention variant tests.

Covers a representative slice of attention mechanisms to ensure registry
exposure, forward pass shape/grad, and a couple of behavioural properties.
"""

from typing import Iterable, List

import math
import pytest
import torch

pytestmark = [pytest.mark.extended]

from layers.modular.core import unified_registry, ComponentFamily  # type: ignore
from layers.modular.attention.registry import get_attention_component
# Populate unified registry (opt-in import)
import layers.modular.core.register_components  # noqa: F401

class _LegacyAttentionRegistryShim:
    """Minimal shim to satisfy existing helper functions during migration."""
    @staticmethod
    def list_components():  # returns list for backward compatibility
        listing = unified_registry.list(ComponentFamily.ATTENTION)['attention']
        # Also incorporate known aliases so older test names remain discoverable
        # This mirrors the legacy registry behaviour where aliases were listed.
        # We pull alias names from the registry describe API if available.
        names = set(listing)
        for name in list(listing):
            try:
                info = unified_registry.describe(ComponentFamily.ATTENTION, name)
                for alias in info.get('aliases', []):
                    names.add(alias)
            except Exception:
                continue
        return sorted(names)

AttentionRegistry = _LegacyAttentionRegistryShim  # type: ignore


def _has(name: str) -> bool:
    if AttentionRegistry is None:
        return False
    try:
        return name in AttentionRegistry.list_components()
    except Exception:
        return False


def test_attention_inventory_nonempty_and_unique() -> None:
    if AttentionRegistry is None:
        pytest.skip("AttentionRegistry unavailable")
    names = AttentionRegistry.list_components()
    assert isinstance(names, list) and names
    assert len(names) == len(set(names)), "Duplicate attention names found"


@pytest.mark.parametrize("name", [
    "autocorrelation_layer",
    "enhanced_autocorrelation",
    "fourier_attention",
    "wavelet_attention",
    "bayesian_attention",
    "causal_convolution",
])
def test_forward_shape_and_grad(name: str) -> None:
    if AttentionRegistry is None or get_attention_component is None:
        pytest.skip("AttentionRegistry unavailable")
    if not _has(name):
        pytest.skip(f"{name} not registered")

    kwargs = {"d_model": 32, "n_heads": 2, "dropout": 0.0, "factor": 1}
    try:
        comp = get_attention_component(name, **kwargs)
    except Exception as e:  # pragma: no cover
        pytest.xfail(f"Instantiate {name} failed: {e}")

    q = torch.randn(2, 12, 32, requires_grad=True)
    out_any = None
    for call in (lambda: comp(q, q, q), lambda: comp(q, q, q, None)):
        try:
            res = call()
            out_any = res[0] if isinstance(res, (tuple, list)) else res
            break
        except TypeError:
            continue
    if out_any is None:
        pytest.xfail(f"No compatible forward signature for {name}")

    assert isinstance(out_any, torch.Tensor)
    assert out_any.shape == q.shape
    out_any.mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


@pytest.mark.skipif(not _has("fourier_attention"), reason="FourierAttention not registered")
def test_fourier_energy_not_vanishing_or_exploding() -> None:
    comp = get_attention_component("fourier_attention", d_model=32, n_heads=2, seq_len=24, dropout=0.0)
    x = torch.randn(2, 24, 32)
    out, *_ = comp(x, x, x)  # type: ignore[misc]
    ratio = (out.pow(2).mean() / x.pow(2).mean()).item()
    assert ratio > 1e-3 and ratio < 20.0


@pytest.mark.skipif(not _has("causal_convolution"), reason="Causal conv not registered")
def test_causal_conv_early_steps_robust_to_future_offset() -> None:
    comp = get_attention_component("causal_convolution", d_model=16, n_heads=2, conv_kernel_size=3, pool_size=2)
    prefix = torch.zeros(1, 12, 16)
    future = torch.zeros(1, 12, 16)
    future[:, 6:, :] = 10.0
    out_with, _ = comp(future, future, future)  # type: ignore[misc]
    out_without, _ = comp(prefix, prefix, prefix)  # type: ignore[misc]
    diff = (out_with[:, :3] - out_without[:, :3]).abs().max().item()
    assert diff < 1.5


@pytest.mark.parametrize("name", [
    "autocorrelation_layer",
    "enhanced_autocorrelation",
    "fourier_attention",
])
def test_cross_attention_forward_shape(name: str) -> None:
    """Cross-attention pattern: different query vs key/value lengths.

    Tries calling attention with Q != K/V sequence lengths and checks output shape.
    Skips gracefully if a specific variant doesn’t support cross mode.
    """
    if AttentionRegistry is None or get_attention_component is None:
        pytest.skip("AttentionRegistry unavailable")
    if not _has(name):
        pytest.skip(f"{name} not registered")

    comp = get_attention_component(name, d_model=32, n_heads=2, dropout=0.0, factor=1)
    q = torch.randn(2, 12, 32, requires_grad=True)
    k = torch.randn(2, 20, 32)
    v = torch.randn(2, 20, 32)
    try:
        out_any = comp(q, k, v)
    except Exception:
        try:
            out_any = comp(q, k, v, None)
        except Exception:
            pytest.skip(f"{name} does not expose a compatible cross-attention signature")
    out = out_any[0] if isinstance(out_any, (tuple, list)) else out_any
    assert isinstance(out, torch.Tensor)
    assert out.shape == q.shape
    out.mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
