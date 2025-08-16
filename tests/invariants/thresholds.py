"""Central threshold configuration for invariant tests.

Starter values are intentionally *slightly lax*; tighten after empirical baselining.
"""
from __future__ import annotations

from typing import Dict


THRESHOLDS: Dict[str, float] = {
    # Decomposition
    "decomposition_recon_rel_err": 1e-4,
    "trend_high_freq_ratio": 0.20,
    "seasonal_mean_abs_ratio": 0.05,
    "decomposition_energy_drift": 0.08,  # tightened from 0.10 after empirical ~0.063 baseline
    # Attention
    "attention_row_sum_tol": 1e-5,
    "attention_mask_leak_max": 1e-6,
    "attention_freq_shift_bins": 1.0,
    "attention_required_lag_hit_rate": 0.6,
    "attention_impulse_focus_min_mass": 0.35,
    "attention_output_norm_ratio_max": 10.0,
    "attention_permutation_recon_rel_err": 1e-5,
    "attention_impulse_focus_hit_rate": 0.5,
    # Encoder
    "encoder_hf_energy_tolerance": 0.05,  # allowable fractional increase layer-to-layer
    "encoder_input_grad_min_norm": 1e-6,
    "encoder_determinism_tol": 1e-7,
    "encoder_hf_increase_tol": 0.05,
}


def get_threshold(key: str) -> float:
    return THRESHOLDS[key]


__all__ = ["THRESHOLDS", "get_threshold"]
