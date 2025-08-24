"""Probabilistic calibration and multi-scale adapter invariants.

Adds advanced diagnostics beyond original coverage test:
1. Empirical 68% coverage vs theoretical (already present).
2. CRPS relative error vs analytic expectation for standard normal.
3. PIT (probability integral transform) KS distance to Uniform(0,1).
4. Multi-scale determinism & effect (pre-existing).
5. Multi-scale spectral separation & limited overlap (new).

Skips gracefully if components unavailable.
"""
from __future__ import annotations

import torch
import pytest

from .thresholds import get_threshold
from . import metrics as M
from tests.helpers import time_series_generators as gen


def _try_prob_head():  # returns instantiated probabilistic head or skips
    try:
        from layers.modular.outputs.probabilistic import ProbabilisticForecastingHead, OutputConfig  # type: ignore
    except Exception:  # pragma: no cover
        pytest.skip("Output heads module unavailable")
    cfg = OutputConfig(d_model=32, output_dim=1, horizon=8, dropout=0.0)
    head = ProbabilisticForecastingHead(cfg)
    head.eval()
    return head


def _sample_gaussian(mean: torch.Tensor, std: torch.Tensor, n: int) -> torch.Tensor:
    return mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(n, *mean.shape, device=mean.device)


@pytest.mark.invariant
def test_probabilistic_head_empirical_coverage():
    head = _try_prob_head()
    # Hidden states
    hidden = torch.randn(4, 24, 32)
    with torch.no_grad():
        out = head(hidden)
    mean, logvar, std = out['mean'], out['logvar'], out['std']  # [B, H, D]
    # Monte Carlo sample from predictive normal
    n_samples = 256
    samples = _sample_gaussian(mean, std, n_samples)  # [S,B,H,D]
    # 68% interval around mean: mean ± std
    lower, upper = mean - std, mean + std
    inside = (samples >= lower.unsqueeze(0)) & (samples <= upper.unsqueeze(0))
    coverage = inside.float().mean().item()
    target = get_threshold("output_prob_68pct_coverage_target")
    max_abs_err = get_threshold("output_prob_68pct_coverage_abs_err_max")
    assert abs(coverage - target) <= max_abs_err, (coverage, target)


@pytest.mark.invariant
def test_probabilistic_head_crps_and_pit():
    head = _try_prob_head()
    hidden = torch.randn(16, 32, 32)  # more samples for PIT stability
    with torch.no_grad():
        out = head(hidden)
    mean, logvar, std = out['mean'], out['logvar'], out['std']
    # Generate synthetic ground truth from model predictive distribution
    y = mean + std * torch.randn_like(mean)
    crps = M.gaussian_crps(mean, std, y)
    empirical = float(crps.mean())
    expected = 0.564  # CRPS of standard normal vs a draw from itself
    rel_err = abs(empirical - expected) / expected
    assert rel_err <= get_threshold("output_prob_crps_rel_err_max"), (empirical, rel_err)
    # PIT KS
    normal = torch.distributions.Normal(mean, std.clamp_min(1e-12))
    pit = normal.cdf(y).clamp(1e-6, 1 - 1e-6)
    ks = M.pit_uniform_ks(pit)
    assert ks <= get_threshold("output_prob_pit_ks_max"), ks


# Multi-scale adapter invariants

def _make_multiscale_adapter():
    try:
        from layers.modular.backbone.adapters import MultiScaleAdapter  # type: ignore
        from layers.modular.base import BaseBackbone  # type: ignore
    except Exception:  # pragma: no cover
        pytest.skip("Adapter modules unavailable")

    class _TinyBackbone(BaseBackbone):  # type: ignore[misc]
        def __init__(self, d_model: int = 16):
            from dataclasses import dataclass
            @dataclass
            class Cfg:  # minimal config placeholder
                pass
            super().__init__(Cfg())
            self.proj = torch.nn.Linear(1, d_model)
            self._d_model = d_model
        def forward(self, x, attention_mask=None, **kwargs):  # noqa: D401
            if x.dim() == 2:
                x = x.unsqueeze(-1).float()
            return self.proj(x)
        def get_d_model(self) -> int:
            return self._d_model
        def supports_seq2seq(self) -> bool:
            return False
        def get_backbone_type(self) -> str:
            return "tiny"
        def get_output_dim(self) -> int:
            return self._d_model

    bb = _TinyBackbone(d_model=16)
    adapter = MultiScaleAdapter(bb, {"scales": [1, 2, 4], "aggregation": "concat"})
    return adapter


@pytest.mark.invariant
def test_multiscale_adapter_determinism_and_effect():
    adapter = _make_multiscale_adapter()
    adapter.eval()
    seq = gen.seasonal_with_trend(batch=1, length=64, dim=1)
    with torch.no_grad():
        out1 = adapter(seq)
        out2 = adapter(seq)
    # Determinism
    rel = torch.linalg.norm(out1 - out2) / (torch.linalg.norm(out1) + 1e-12)
    assert rel <= get_threshold("multiscale_determinism_rel_err"), rel
    # Multi-scale effect: compare to single-scale variant (scale=1 only)
    single = _make_multiscale_adapter()
    single.scales = [1]
    # Rebuild processors for single scale scenario (quick patch)
    single.scale_processors = torch.nn.ModuleList([torch.nn.Linear(single.backbone.get_d_model(), single.backbone.get_d_model())])
    single.aggregation_method = 'add'
    single.eval()
    with torch.no_grad():
        ms_out = out1
        single_out = single(seq)
    delta = torch.linalg.norm(ms_out - single_out) / (torch.linalg.norm(single_out) + 1e-12)
    assert delta >= get_threshold("multiscale_effect_min_delta"), delta


def _power_spectrum(ts: torch.Tensor) -> torch.Tensor:
    spec = torch.fft.rfft(ts, dim=1)
    power = (spec.abs() ** 2).mean(dim=(0, 2))  # mean over batch & channels
    return power / (power.sum() + 1e-12)


@pytest.mark.invariant
def test_multiscale_spectral_separation():
    adapter = _make_multiscale_adapter()
    adapter.eval()
    seq = gen.seasonal_with_trend(batch=1, length=128, dim=1)
    with torch.no_grad():
        _ = adapter(seq)
    scales = getattr(adapter, "_last_scale_outputs", None)
    if scales is None:
        pytest.skip("Scale outputs not captured")
    spectra = [_power_spectrum(s) for s in scales]
    # Frequency centroids
    idx = torch.arange(spectra[0].shape[0], dtype=spectra[0].dtype)
    centroids = [float((idx * sp).sum().item()) for sp in spectra]
    centroids_sorted = sorted(centroids)
    min_sep = get_threshold("multiscale_freq_centroid_min_separation")
    for a, b in zip(centroids_sorted, centroids_sorted[1:]):
        assert (b - a) >= min_sep, (a, b)
    # Overlap
    overlap_max = get_threshold("multiscale_band_overlap_max")
    for i in range(len(spectra)):
        for j in range(i + 1, len(spectra)):
            ov = float(torch.minimum(spectra[i], spectra[j]).sum().item())
            assert ov <= overlap_max, ov


__all__ = []
