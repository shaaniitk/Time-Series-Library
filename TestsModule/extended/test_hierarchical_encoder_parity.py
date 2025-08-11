"""Semantic parity tests for hierarchical encoder.

Checks that hierarchical encoder roughly preserves:
1. Output mean close to base encoder output mean (same config single layer).
2. Relative ordering of variance across batch items.
3. Gradient norm scale within tolerance versus standard encoder.
"""
from __future__ import annotations

import pytest
import torch
from TestsModule.utils import make_encoder

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from layers.modular.encoder import get_encoder_component  # type: ignore
except Exception:  # pragma: no cover
    get_encoder_component = None  # type: ignore


@pytest.mark.skipif(get_encoder_component is None, reason="Encoder registry unavailable")
def test_hierarchical_semantic_parity_basic() -> None:
    d_model = 32
    n_heads = 4
    seq_len = 24
    x = torch.randn(2, seq_len, d_model, requires_grad=True)

    std_enc = make_encoder("standard", d_model=d_model, n_heads=n_heads)
    hier_enc = make_encoder("hierarchical", d_model=d_model, n_heads=n_heads)

    std_out, _ = std_enc(x)
    hier_out, _ = hier_enc(x)
    assert std_out.shape == hier_out.shape == x.shape

    mean_diff = (std_out.mean() - hier_out.mean()).abs().item()
    # Means should be within a loose tolerance (encoders may differ but not explode)
    assert mean_diff < 0.5, f"Hierarchical mean deviates too much ({mean_diff:.3f})"

    # Variance ordering per batch item
    std_vars = std_out.var(dim=(1, 2))
    hier_vars = hier_out.var(dim=(1, 2))
    order_std = torch.argsort(std_vars)
    order_hier = torch.argsort(hier_vars)
    assert torch.equal(order_std, order_hier), "Relative per-sample variance ordering changed"

    # Gradient norm parity
    (std_out.sum()).backward(retain_graph=True)
    grad_std = x.grad.detach().clone()
    x.grad.zero_()
    (hier_out.sum()).backward()
    grad_hier = x.grad.detach()
    ratio = (grad_hier.norm() / (grad_std.norm() + 1e-6)).item()
    assert 0.2 <= ratio <= 5.0, f"Gradient norm ratio out of bounds: {ratio:.3f}"
