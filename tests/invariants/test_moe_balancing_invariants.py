"""Mixture-of-Experts (MoE) usage balancing invariants (Phase 4 - Step 5).

Ensures that routing does not collapse onto a single expert and that average
expert utilization over a small batch meets minimal diversity criteria.

Skips gracefully if MoE implementation unavailable.
"""
from __future__ import annotations

import pytest
import torch

from .thresholds import get_threshold


def _build_moe():  # type: ignore[return-type]
    try:
        from layers.GatedMoEFFN import GatedMoEFFN  # type: ignore
    except Exception:  # pragma: no cover
        pytest.skip("MoE layer unavailable")
    torch.manual_seed(0)
    return GatedMoEFFN(d_model=32, d_ff=64, num_experts=4)


@pytest.mark.invariant
def test_moe_expert_usage_balanced():  # type: ignore
    module = _build_moe()
    module.eval()
    # Generate a modest batch to exercise routing diversity
    x = torch.randn(6, 48, 32)
    with torch.no_grad():
        out, aux_loss = module(x)
    assert out.shape == x.shape
    # Access internal gating stats if present
    if not hasattr(module, 'gating_network'):
        pytest.skip('No gating attribute on module')
    gating = module.gating_network
    flat = x.view(-1, x.size(-1))
    with torch.no_grad():
        topk_indices, combined_weights, soft_weights, aux_loss = gating(flat)  # type: ignore
    counts = torch.bincount(topk_indices.view(-1), minlength=module.num_experts).float()
    total = counts.sum().item() + 1e-12
    usage_frac = counts / total
    non_zero = usage_frac[usage_frac > 0]
    min_mean_frac = get_threshold("moe_min_mean_usage_frac")
    imbalance_ratio_max = get_threshold("moe_max_usage_imbalance_ratio")
    assert usage_frac.mean().item() >= min_mean_frac / 2.0, usage_frac.tolist()
    assert (usage_frac >= min_mean_frac).sum().item() >= 2, usage_frac.tolist()
    if non_zero.numel() >= 2:
        ratio = (non_zero.max() / (non_zero.min() + 1e-12)).item()
        assert ratio <= imbalance_ratio_max, (ratio, usage_frac.tolist())
    if aux_loss is not None:
        if isinstance(aux_loss, torch.Tensor):
            assert torch.isfinite(aux_loss).all(), "Aux loss not finite"
        else:
            assert isinstance(aux_loss, (float, int)), "Aux loss unexpected type"

__all__ = []
