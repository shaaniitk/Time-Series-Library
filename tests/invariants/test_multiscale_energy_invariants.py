"""Multi-scale adapter energy decomposition invariants.

Checks:
1. Energy conservation (reconstruction): sum of per-scale outputs (after linear alignment) approximates final aggregated output.
2. Non-collapse: each scale (except possibly scale=1) contributes a minimum fraction of total energy.
3. Non-domination: no single scale accounts for nearly all energy.

Skips gracefully if MultiScaleAdapter unavailable.
"""
from __future__ import annotations

import pytest
import torch

from .thresholds import get_threshold
from tests.helpers import time_series_generators as gen


def _build_adapter(scales=(1,2,4)):
    try:
        from utils.modular_components.implementations.adapters import MultiScaleAdapter  # type: ignore
        from utils.modular_components.base_interfaces import BaseBackbone  # type: ignore
    except Exception:  # pragma: no cover
        pytest.skip("Adapter modules unavailable")

    class _TinyBackbone(BaseBackbone):  # type: ignore[misc]
        def __init__(self, d_model: int = 16):
            from dataclasses import dataclass
            @dataclass
            class Cfg: pass
            super().__init__(Cfg())
            self.proj = torch.nn.Linear(1, d_model)
            self._d_model = d_model
        def forward(self, x, attention_mask=None, **kwargs):
            if x.dim()==2:
                x = x.unsqueeze(-1).float()
            return self.proj(x)
        def get_d_model(self): return self._d_model
        def supports_seq2seq(self): return False
        def get_backbone_type(self): return "tiny"
        def get_output_dim(self): return self._d_model

    bb = _TinyBackbone(16)
    return MultiScaleAdapter(bb, {"scales": list(scales), "aggregation": "concat"})


@pytest.mark.invariant
def test_multiscale_energy_decomposition():
    adapter = _build_adapter()
    adapter.eval()
    seq = gen.seasonal_with_trend(batch=1, length=96, dim=1)
    with torch.no_grad():
        final = adapter(seq)  # triggers _last_scale_outputs capture
    assert hasattr(adapter, "_last_scale_outputs"), "Per-scale outputs not captured"
    scales = adapter._last_scale_outputs  # type: ignore[attr-defined]
    # All per-scale tensors share shape [B,L,D]
    # Reconstruction: simple mean of concatenated parts after linear down-projection used in implementation.
    # Because adapter aggregates via concat->linear, we approximate reconstruction by linear regressing from
    # concatenated scale outputs to final and measuring relative error after solving least squares once.
    concat = torch.cat(scales, dim=-1)  # [B,L, D*num_scales]
    B,L,_ = concat.shape
    X = concat.view(B*L, -1)
    Y = final.view(B*L, -1)
    # Solve (X^T X) w = X^T Y
    XtX = X.T @ X + 1e-6*torch.eye(X.size(1))
    w = torch.linalg.solve(XtX, X.T @ Y)
    recon = (X @ w).view_as(final)
    recon_err = torch.linalg.norm(recon - final) / (torch.linalg.norm(final) + 1e-12)
    assert recon_err <= get_threshold("multiscale_energy_reconstruction_rel_err"), recon_err

    # Energy per scale
    energies = torch.tensor([torch.linalg.norm(s).item()**2 for s in scales])
    total_energy = energies.sum().item() + 1e-12
    fracs = energies / total_energy

    min_frac = get_threshold("multiscale_min_unique_scale_energy_frac")
    max_dom = get_threshold("multiscale_max_dominant_scale_energy_frac")

    # Non-collapse: at least two scales exceed min_frac (allow base scale to dominate slightly if others small)
    assert (fracs >= min_frac).sum().item() >= 2, fracs.tolist()
    # Non-domination
    assert fracs.max().item() <= max_dom, fracs.tolist()

__all__ = []
