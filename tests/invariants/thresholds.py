"""Central threshold configuration for invariant tests.

Starter values are intentionally *slightly lax*; tighten after empirical baselining.
"""
from __future__ import annotations

from typing import Dict


THRESHOLDS: Dict[str, float] = {
    # Decomposition
    "decomposition_recon_rel_err": 5e-6,
    "trend_high_freq_ratio": 0.12,
    "seasonal_mean_abs_ratio": 0.02,
    "decomposition_energy_drift": 0.065,  # further tightened (empirical ~0.063 baseline)
    # Attention
    "attention_row_sum_tol": 1e-6,
    "attention_mask_leak_max": 1e-6,
    "attention_freq_shift_bins": 0.5,
    "attention_required_lag_hit_rate": 0.9,
    "attention_impulse_focus_min_mass": 0.80,
    "attention_output_norm_ratio_max": 2.5,
    "attention_permutation_recon_rel_err": 5e-6,
    "attention_impulse_focus_hit_rate": 0.8,
    # Encoder
    "encoder_hf_energy_tolerance": 0.02,  # tightened allowable fractional increase layer-to-layer
    "encoder_input_grad_min_norm": 1e-6,
    "encoder_determinism_tol": 5e-8,
    "encoder_hf_increase_tol": 0.05,
    # Adapter
    # Allow exact zero; downstream test uses <= semantics (keep tiny epsilon)
    "adapter_pass_through_rel_err": 5e-9,
    "adapter_covariate_effect_min_delta": 0.117514,
    "adapter_min_effective_rank_fraction": 0.6,
    # Embedding
    "embedding_positional_effect_min_delta": 1e-6,  # unchanged (already very small)
    "embedding_temporal_feature_effect_min_delta": 5e-5,
    "embedding_token_norm_max": 25.0,  # tightened upper bound
    # Output Heads
    "output_head_determinism_rel_err": 5e-8,
    "output_prob_logvar_min": -8.000001,
    "output_prob_logvar_max": 8.000001,
    "output_prob_std_consistency_rel_err": 5e-8,
    # Probabilistic calibration
    "output_prob_68pct_coverage_target": 0.6827,  # theoretical normal 1-sigma coverage
    "output_prob_68pct_coverage_abs_err_max": 0.12,
    # Multi-scale adapter
    "multiscale_effect_min_delta": 5e-5,
    "multiscale_determinism_rel_err": 5e-8,
    # Multi-scale energy decomposition
    "multiscale_min_unique_scale_energy_frac": 0.05,  # each non-trivial scale contributes at least 5%
    "multiscale_max_dominant_scale_energy_frac": 0.90,  # no single scale should dominate >90%
    # Empirical baseline ~4.75e-05; keep margin
    "multiscale_energy_reconstruction_rel_err": 5e-05,
    # Multi-scale spectral separation
    "multiscale_freq_centroid_min_separation": 0.05,
    # Tightened overlap after initial pass (empirical ~0.83 baseline simple linear processors)
    "multiscale_band_overlap_max": 0.98,
    # Probabilistic advanced diagnostics
    "output_prob_crps_rel_err_max": 0.15,
    "output_prob_pit_ks_max": 0.25,
    # Quantile forecasting
    "quantile_monotonic_violation_max_frac": 0.0,  # expect zero violations after enforcement
    "quantile_coverage_abs_err_max": 0.12,  # same laxity as 68% interval initial
    # MoE balancing & auxiliary metrics (initial values; refine with tuner)
    "moe_min_mean_usage_frac": 0.05,
    "moe_max_usage_imbalance_ratio": 25.0,  # max / min usage (non-zero) upper bound
    "moe_load_balance_loss_max": 5.0,       # very lax upper bound (expect <<1 normally)
    "moe_entropy_min": 0.1,                # minimum routing entropy (encourage diversity)
    # Embedding / Positional
    "embedding_positional_norm_var_max": 1e-4,
    "embedding_positional_value_abs_max": 1.001,
    "embedding_slice_consistency_rel_err": 5e-7,
    "embedding_positional_offdiag_mean_abs_cos_max": 0.85,
}


def get_threshold(key: str) -> float:
    return THRESHOLDS[key]


__all__ = ["THRESHOLDS", "get_threshold"]
