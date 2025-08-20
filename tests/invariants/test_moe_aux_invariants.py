"""MoE auxiliary metric invariants.

Validates routing entropy and load-balance auxiliary loss remain within
configured bounds. Skips if MoE implementation missing.
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
def test_moe_aux_metrics_entropy_and_balance():  # type: ignore
    module = _build_moe()
    module.train()
    x = torch.randn(4, 32, 32)
    with torch.no_grad():
        _, _ = module(x)
    if not hasattr(module, 'gating_network'):
        pytest.skip('No gating on module')
    gating = module.gating_network
    flat = x.view(-1, x.size(-1))
    with torch.no_grad():
        topk_indices, combined_weights, soft_weights, aux_loss = gating(flat)  # type: ignore
    # soft_weights shape [N, E]; compute entropy per token then mean
    if soft_weights is not None:
        p = soft_weights / (soft_weights.sum(dim=-1, keepdim=True) + 1e-12)
        entropy = -(p * (p + 1e-12).log()).sum(dim=-1).mean().item()
        assert entropy >= get_threshold("moe_entropy_min"), entropy
    if aux_loss is not None:
        # assume aux_loss encodes load balance; treat magnitude as must remain below threshold
        if isinstance(aux_loss, torch.Tensor):
            val = float(aux_loss.item())
            assert val <= get_threshold("moe_load_balance_loss_max"), val

__all__ = []
