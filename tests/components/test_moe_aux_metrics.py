"""Focused tests for Mixture-of-Experts (MoE) FFN auxiliary metrics.

Validates that exposed auxiliary losses / metrics remain finite and shaped correctly.
"""
from __future__ import annotations

import pytest
import torch

try:
    from layers.modular.feedforward.feedforward import MoEFFN  # type: ignore
except Exception:  # pragma: no cover
    MoEFFN = None  # type: ignore

@pytest.mark.smoke
def test_moe_auxiliary_metrics_basic() -> None:
    """Run a forward pass through MoEFFN and inspect aux metrics."""
    if MoEFFN is None:
        pytest.skip("MoEFFN unavailable")

    torch.manual_seed(0)
    module = MoEFFN(d_model=32, d_ff=64, n_experts=4, top_k=2)

    x = torch.randn(3, 10, 32)
    out = module(x)
    assert out.shape == x.shape

    aux = module.get_auxiliary_losses()
    # Expected keys (may evolve but keep core ones stable)
    for key in ["load_balance_loss", "entropy_loss", "expert_usage"]:
        assert key in aux, f"Missing aux metric: {key}"

    # Basic finiteness checks
    if aux["load_balance_loss"] is not None:
        lbl = aux["load_balance_loss"]
        assert torch.isfinite(lbl), "Non-finite load balance loss"
    if aux["entropy_loss"] is not None:
        ent = aux["entropy_loss"]
        assert torch.isfinite(ent), "Non-finite entropy loss"

    usage = aux.get("expert_usage")
    if usage is not None:
        assert usage.numel() == module.n_experts
        assert torch.isfinite(usage).all()
