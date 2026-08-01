import hashlib
import json
import math
import os
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path

import torch


TFT_LEGACY_EXTENSION_SEMANTICS_VERSION = 1
TFT_CURRENT_EXTENSION_SEMANTICS_VERSION = 2
TFT_SUPPORTED_EXTENSION_SEMANTICS_VERSIONS = (
    TFT_LEGACY_EXTENSION_SEMANTICS_VERSION,
    TFT_CURRENT_EXTENSION_SEMANTICS_VERSION,
)
TFT_CHECKPOINT_METADATA_FILENAME = "checkpoint_metadata.json"
TFT_RESULT_METADATA_FILENAME = "tft_result_metadata.json"


TFT_EXTENSION_MIGRATION_CAPABILITIES = {
    "core": {
        "config_flag": None,
        "v1_replay": "supported",
        "v1_to_v2": "retrain",
        "v2_artifact_status": "released",
        "reason": "SR00 conservatively forbids every cross-version weight load",
    },
    "fft_branch": {
        "config_flag": "tft_use_fft_branch",
        "v1_replay": "semantics_v1_only",
        "v1_to_v2": "retrain",
        "repair_task": "TFT-SR03",
        "v2_artifact_status": "released",
        "reason": "v2 changes filter/selection and fusion semantics",
    },
    "explicit_cross_attention": {
        "config_flag": "tft_use_explicit_cross_attention",
        "v1_replay": "semantics_v1_only",
        "v1_to_v2": "retrain",
        "repair_task": "TFT-SR04",
        "v2_artifact_status": "released",
        "reason": "v2 changes neutral integration and interpretation semantics",
    },
    "lag_attention": {
        "config_flag": "tft_use_lag_attention",
        "v1_replay": "semantics_v1_only",
        "v1_to_v2": "retrain",
        "repair_task": "TFT-SR05",
        "v2_artifact_status": "released",
        "reason": "v2 separates prefix, exact-token, and calendar-time lag semantics",
    },
    "higher_order_interaction": {
        "config_flag": "tft_use_higher_order",
        "v1_replay": "semantics_v1_only",
        "v1_to_v2": "retrain",
        "repair_task": "TFT-SR06",
        "v2_artifact_status": "released",
        "reason": "v2 separates latent polynomial and named feature interactions",
    },
    "per_feature_vsn": {
        "config_flag": "tft_vsn_per_feature_gating",
        "v1_replay": "unsupported_defective_path",
        "v1_to_v2": "retrain",
        "repair_task": "TFT-SR06",
        "v2_artifact_status": "released",
        "reason": "v1 per-feature gating is a reproduced defective path",
    },
    "temporal_compression": {
        "config_flag": "tft_use_temporal_compression",
        "v1_replay": "semantics_v1_only",
        "v1_to_v2": "retrain",
        "repair_task": "TFT-SR07",
        "v2_artifact_status": "released",
        "reason": "v2 replaces dead codec behavior with live memory compression",
    },
    "graph_cross_mixing": {
        "config_flag": "tft_cross_variable_mixing",
        "v1_replay": "semantics_v1_only",
        "v1_to_v2": "retrain",
        "repair_task": "TFT-SR08",
        "v2_artifact_status": "released",
        "reason": "v2 changes graph typing, heads, density, and identity insertion",
    },
    "covariate_reattention": {
        "config_flag": "tft_covariate_reattention",
        "v1_replay": "semantics_v1_only",
        "v1_to_v2": "retrain",
        "repair_task": "TFT-SR04",
        "v2_artifact_status": "pending_repair",
        "reason": "v2 comparisons require neutral integration and paired state",
    },
    "regime_moe": {
        "config_flag": "tft_use_regime_moe",
        "v1_replay": "semantics_v1_only",
        "v1_to_v2": "retrain",
        "repair_task": "TFT-SR02",
        "v2_artifact_status": "released",
        "reason": "v2 retains the base feed-forward path and neutralizes both the MoE residual and auxiliary objective",
    },
    "dual_attention_fusion": {
        "config_flag": "tft_dual_attention_fusion",
        "v1_replay": "semantics_v1_only",
        "v1_to_v2": "retrain",
        "repair_task": "TFT-SR02",
        "v2_artifact_status": "released",
        "reason": "v2 keeps the declared base attention branch at unit strength and adds supplementary attention residually",
    },
    "vsn_residual_bypass": {
        "config_flag": "tft_vsn_residual_bypass",
        "v1_replay": "semantics_v1_only",
        "v1_to_v2": "retrain",
        "repair_task": "TFT-SR02",
        "v2_artifact_status": "released",
        "reason": "the existing direct tanh gate is already an exact zero residual; SR02 only harmonizes its adapter diagnostics and knockout API",
    },
}


TFT_EXTENSION_INTEGRATION_MODES = (
    "off",
    "neutral",
    "small_residual",
    "legacy",
)
TFT_EXTENSION_RESIDUAL_SHAPES = ("scalar", "channel")
TFT_POSITION_UNITS = ("steps", "trading_sessions", "calendar_days")
TFT_POSITION_SOURCES = ("row_index", "explicit_argument", "known_feature")

# Built-in ETT loaders carry a fixed-frequency dataset contract.  Custom and
# market datasets are intentionally excluded because weekends, holidays, and
# missing observations make row spacing a modelling choice, not a fact.
TFT_REGULAR_SAMPLING_DATASETS = frozenset({"ETTh1", "ETTh2", "ETTm1", "ETTm2"})


# This registry is the single public map from an optional additive/structural
# branch to its activation flag and resolved integration mode.  The mode fields
# are semantics-v2 configuration and deliberately do not belong to
# TFT_BASE_DEFAULTS, whose version-1 digest material is frozen below.
TFT_EXTENSION_INTEGRATION_SPECS = {
    "fft_branch": {
        "config_flag": "tft_use_fft_branch",
        "mode_key": "tft_fft_integration_mode",
        "integration_kind": "additive",
    },
    "explicit_cross_attention": {
        "config_flag": "tft_use_explicit_cross_attention",
        "mode_key": "tft_cross_attention_integration_mode",
        "integration_kind": "additive",
    },
    "lag_attention": {
        "config_flag": "tft_use_lag_attention",
        "mode_key": "tft_lag_integration_mode",
        "integration_kind": "additive",
    },
    "higher_order_interaction": {
        "config_flag": "tft_use_higher_order",
        "mode_key": "tft_higher_order_integration_mode",
        "integration_kind": "additive",
    },
    "temporal_compression": {
        "config_flag": "tft_use_temporal_compression",
        "mode_key": "tft_temporal_compression_integration_mode",
        "integration_kind": "pending_structural_repair",
    },
    "graph_cross_mixing": {
        "config_flag": "tft_cross_variable_mixing",
        "mode_key": "tft_graph_integration_mode",
        "integration_kind": "additive",
    },
    "covariate_reattention": {
        "config_flag": "tft_covariate_reattention",
        "mode_key": "tft_covariate_reattention_integration_mode",
        "integration_kind": "additive",
    },
    "regime_moe": {
        "config_flag": "tft_use_regime_moe",
        "mode_key": "tft_regime_moe_integration_mode",
        "integration_kind": "additive",
    },
    "dual_attention_fusion": {
        "config_flag": "tft_dual_attention_fusion",
        "mode_key": "tft_dual_attention_integration_mode",
        "integration_kind": "additive",
    },
    "vsn_residual_bypass": {
        "config_flag": "tft_vsn_residual_bypass",
        "mode_key": "tft_vsn_bypass_integration_mode",
        "integration_kind": "additive",
    },
}


for _extension_name, _integration_spec in TFT_EXTENSION_INTEGRATION_SPECS.items():
    _capability = TFT_EXTENSION_MIGRATION_CAPABILITIES[_extension_name]
    if _capability["config_flag"] != _integration_spec["config_flag"]:
        raise RuntimeError(
            f"TFT extension registry mismatch for {_extension_name}: "
            "activation flags disagree."
        )
    _capability["integration_mode_key"] = _integration_spec["mode_key"]
    _capability["integration_kind"] = _integration_spec["integration_kind"]
del _extension_name, _integration_spec, _capability


TFT_V2_ONLY_DEFAULTS = {
    **{
        spec["mode_key"]: None
        for spec in TFT_EXTENSION_INTEGRATION_SPECS.values()
    },
    "tft_extension_residual_shape": "scalar",
    "tft_small_residual_init": 1e-3,
    "tft_position_unit": "steps",
    "tft_position_source": "row_index",
    "tft_position_feature_name": None,
    "tft_declared_regular_sampling": False,
    "tft_regular_sampling_declaration_source": None,
    "tft_lag_semantics_mode": "shifted_prefix_attention",
    "tft_temporal_compression_mode": "kv_pool",
    "tft_tc_min_long_sequence": 512,
    "tft_tc_experimental_short_window": False,
    "tft_graph_history_top_k": None,
    "tft_graph_future_top_k": None,
    "tft_graph_history_density": None,
    "tft_graph_future_density": None,
    "tft_graph_self_edge_policy": "allowed",
    "tft_graph_head_mode": "single",
    "tft_graph_temperature": 1.0,
    "tft_graph_entropy_regularization": 0.0,
    "tft_graph_support_stability_regularization": 0.0,
    "tft_graph_residual_strength_init": 0.0,
    "tft_graph_scope": "observed_and_known",
}


TFT_BASE_DEFAULTS = {
    "tft_profile": "extended_safe",
    "tft_extension_semantics_version": TFT_CURRENT_EXTENSION_SEMANTICS_VERSION,
    "tft_use_swiglu": False,
    "tft_full_attention": False,
    "tft_cross_variable_mixing": False,
    "tft_allow_custom_known": False,
    "tft_known_len": None,
    "tft_known_max_channels": 512,
    "tft_known_feature_names": None,
    "tft_vsn_residual_bypass": False,
    "tft_dual_attention_fusion": False,
    "tft_use_lag_attention": False,
    "tft_lag_scales": [1, 2, 4],
    "tft_temporal_backbone": "hybrid_tcn_lstm",
    "tft_temporal_backbone_layers": 3,
    "tft_temporal_kernel_size": 3,
    "tft_temporal_hidden_size": None,
    "tft_use_higher_order": False,
    "tft_interaction_order": 2,
    "tft_interaction_rank": None,
    "tft_use_regime_moe": False,
    "tft_use_explicit_cross_attention": False,
    "tft_cross_attention_type": "full",
    "tft_attention_position_bias": "none",
    "tft_attention_backend": "exact",
    "tft_attention_dropout": 0.0,
    "tft_rope_base": 10000.0,
    "tft_alibi_scale": 1.0,
    "tft_use_revin": False,
    "tft_revin_affine": False,
    "tft_use_quantile_head": False,
    "tft_output_quantiles": [0.1, 0.5, 0.9],
    "tft_output_mode": None,
    "tft_point_loss_coeff": 1.0,
    "tft_quantile_loss_coeff": 1.0,
    "tft_num_regimes": 4,
    "tft_num_moe_experts": 4,
    "tft_moe_top_k": 2,
    "tft_moe_hidden_size": None,
    "tft_moe_noise_epsilon": 1e-2,
    "tft_moe_aux_loss_coeff": 0.0,
    "tft_payload_stack_layers": False,
    "tft_use_fft_branch": False,
    "tft_fft_modes": 32,
    "tft_fft_mode_select": "low",
    "tft_stochastic_depth_rate": 0.0,
    "tft_gradient_checkpointing": False,
    "tft_use_temporal_compression": False,
    "tft_tc_stride": 2,
    "tft_tc_threshold": 256,
    "tft_vsn_n_selection_heads": 1,
    "tft_mlp_quantile_projection": False,
    "tft_quantile_projection_ff_size": 0,
    "tft_per_target_heads": False,
    "tft_vsn_per_feature_gating": False,
    "tft_covariate_reattention": False,
    "tft_moe_capacity_factor": 1.25,
    "tft_vsn_low_rank_threshold": 64,
    "tft_debug_checks": False,
    "tft_graph_type": "dense",
    "tft_graph_top_k": 10,
    "tft_graph_num_layers": 2,
    "tft_graph_temporal_evolution": False,
    "tft_graph_edge_features": False,
}


# Do not append semantics-v2 fields to this material.  The tuple captures the
# historical version-1 digest schema; new repaired controls belong in
# TFT_V2_ONLY_DEFAULTS so the known legacy digest remains byte-for-byte stable.
TFT_LEGACY_V1_DIGEST_KEYS = tuple(
    sorted(
        key
        for key in TFT_BASE_DEFAULTS
        if key != "tft_extension_semantics_version"
    )
)


# Only model-computation/state controls may differ between the discarded
# initialization reference and its paired variant. Dataset/schema, migration,
# debugging, interpretation-only, execution-only, and loss-weight controls are
# excluded so a "paired" comparison cannot change the physical variables or
# supervision contract behind identical tensor shapes.
TFT_PAIRED_ARCHITECTURE_KEYS = frozenset(
    {
        "tft_use_swiglu",
        "tft_full_attention",
        "tft_cross_variable_mixing",
        "tft_vsn_residual_bypass",
        "tft_dual_attention_fusion",
        "tft_use_lag_attention",
        "tft_lag_scales",
        "tft_temporal_backbone",
        "tft_temporal_backbone_layers",
        "tft_temporal_kernel_size",
        "tft_temporal_hidden_size",
        "tft_use_higher_order",
        "tft_interaction_order",
        "tft_interaction_rank",
        "tft_use_regime_moe",
        "tft_num_regimes",
        "tft_num_moe_experts",
        "tft_moe_top_k",
        "tft_moe_hidden_size",
        "tft_moe_noise_epsilon",
        "tft_moe_capacity_factor",
        "tft_use_explicit_cross_attention",
        "tft_cross_attention_type",
        "tft_attention_position_bias",
        "tft_attention_backend",
        "tft_attention_dropout",
        "tft_rope_base",
        "tft_alibi_scale",
        "tft_use_revin",
        "tft_revin_affine",
        "tft_use_quantile_head",
        "tft_output_quantiles",
        "tft_output_mode",
        "tft_use_fft_branch",
        "tft_fft_modes",
        "tft_fft_mode_select",
        "tft_stochastic_depth_rate",
        "tft_use_temporal_compression",
        "tft_tc_stride",
        "tft_tc_threshold",
        "tft_vsn_n_selection_heads",
        "tft_mlp_quantile_projection",
        "tft_quantile_projection_ff_size",
        "tft_per_target_heads",
        "tft_vsn_per_feature_gating",
        "tft_covariate_reattention",
        "tft_vsn_low_rank_threshold",
        "tft_graph_type",
        "tft_graph_top_k",
        "tft_graph_num_layers",
        "tft_graph_temporal_evolution",
        "tft_graph_edge_features",
    }
)


TFT_PROFILE_DEFAULTS = {
    "canonical": {
        "tft_temporal_backbone": "lstm",
        "tft_use_quantile_head": True,
        "tft_output_mode": "joint",
        "tft_use_swiglu": False,
        "tft_full_attention": False,
        "tft_cross_variable_mixing": False,
        "tft_vsn_residual_bypass": False,
        "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False,
        "tft_use_higher_order": False,
        "tft_use_regime_moe": False,
        "tft_use_explicit_cross_attention": False,
        "tft_attention_position_bias": "none",
        "tft_attention_backend": "exact",
        "tft_use_revin": False,
        "tft_revin_affine": False,
        "tft_use_fft_branch": False,
        "tft_use_temporal_compression": False,
        "tft_vsn_n_selection_heads": 1,
        "tft_mlp_quantile_projection": False,
        "tft_per_target_heads": False,
        "tft_vsn_per_feature_gating": False,
        "tft_covariate_reattention": False,
        "tft_graph_type": "dense",
        "tft_graph_temporal_evolution": False,
        "tft_graph_edge_features": False,
        "tft_payload_stack_layers": False,
    },
    "extended_safe": {
        "tft_temporal_backbone": "hybrid_tcn_lstm",
        "tft_use_quantile_head": False,
        "tft_vsn_residual_bypass": False,
        "tft_vsn_per_feature_gating": False,
        "tft_graph_type": "dense",
        "tft_use_fft_branch": False,
        "tft_use_higher_order": False,
        "tft_use_regime_moe": False,
        "tft_use_lag_attention": False,
        "tft_use_explicit_cross_attention": False,
        "tft_covariate_reattention": False,
    },
    "experimental_full": {
        "tft_temporal_backbone": "hybrid_tcn_lstm",
        "tft_use_quantile_head": True,
        "tft_output_mode": "joint",
        "tft_use_swiglu": True,
        "tft_full_attention": True,
        "tft_cross_variable_mixing": True,
        "tft_vsn_residual_bypass": True,
        "tft_dual_attention_fusion": True,
        "tft_use_lag_attention": True,
        "tft_use_higher_order": True,
        "tft_interaction_order": 2,
        "tft_use_regime_moe": True,
        "tft_use_explicit_cross_attention": True,
        "tft_attention_position_bias": "rope",
        "tft_attention_backend": "exact",
        "tft_use_revin": True,
        "tft_revin_affine": True,
        "tft_use_fft_branch": True,
        "tft_fft_mode_select": "learned",
        "tft_use_temporal_compression": True,
        "tft_per_target_heads": True,
        "tft_covariate_reattention": True,
        "tft_graph_type": "sparse",
        "tft_payload_stack_layers": True,
    },
}


TFT_CANONICAL_REQUIRED = dict(TFT_PROFILE_DEFAULTS["canonical"])
TFT_IGNORED_MODEL_KNOBS = {
    "d_ff": 2048,
}


TFT_LEGACY_TO_CANONICAL_FFT_MODE = {
    "low": "low_k",
    "top_amplitude": "top_amplitude_k",
    "learned": "learned_filter",
}


def _normalize_profile_name(value):
    profile = "extended_safe" if value is None else str(value).strip().lower()
    if profile == "":
        profile = "extended_safe"
    if profile not in TFT_PROFILE_DEFAULTS:
        raise ValueError(
            f"tft_profile must be one of {sorted(TFT_PROFILE_DEFAULTS.keys())}, got {profile!r}."
        )
    return profile


def _setdefault_attr(args, key, value):
    if not hasattr(args, key):
        setattr(args, key, deepcopy(value))
        return
    current = getattr(args, key)
    if current is None or current == "":
        setattr(args, key, deepcopy(value))


def _maybe_apply_profile_override(args, key, profile_value):
    current = getattr(args, key, None)
    base_value = TFT_BASE_DEFAULTS.get(key, None)
    if current is None or current == "" or current == base_value:
        setattr(args, key, deepcopy(profile_value))


def _normalize_semantics_version(value):
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "tft_extension_semantics_version must be integer 1 (legacy) or 2 (current)."
        ) from exc
    if version not in TFT_SUPPORTED_EXTENSION_SEMANTICS_VERSIONS:
        raise ValueError(
            "tft_extension_semantics_version must be one of "
            f"{list(TFT_SUPPORTED_EXTENSION_SEMANTICS_VERSIONS)}, got {version!r}."
        )
    return version


def _normalize_fft_mode_select(args):
    raw_value = getattr(args, "tft_fft_mode_select", "low")
    if raw_value is None or raw_value == "":
        raw_value = "low"
    mode = str(raw_value).strip().lower()
    canonical_modes = set(TFT_LEGACY_TO_CANONICAL_FFT_MODE.values())
    valid_modes = set(TFT_LEGACY_TO_CANONICAL_FFT_MODE) | canonical_modes
    if mode not in valid_modes:
        raise ValueError(
            "tft_fft_mode_select must be one of "
            f"{sorted(valid_modes)}, got {raw_value!r}."
        )

    semantics_version = _normalize_semantics_version(
        getattr(
            args,
            "tft_extension_semantics_version",
            TFT_CURRENT_EXTENSION_SEMANTICS_VERSION,
        )
    )
    if semantics_version == TFT_LEGACY_EXTENSION_SEMANTICS_VERSION:
        if mode in canonical_modes:
            inverse = {
                canonical: legacy
                for legacy, canonical in TFT_LEGACY_TO_CANONICAL_FFT_MODE.items()
            }
            mode = inverse[mode]
        args.tft_fft_mode_select = mode
        return mode

    canonical = TFT_LEGACY_TO_CANONICAL_FFT_MODE.get(mode, mode)
    if mode != canonical:
        warnings.warn(
            "Legacy TFT FFT mode name "
            f"{mode!r} is deprecated in semantics v2; use {canonical!r}.",
            DeprecationWarning,
            stacklevel=3,
        )
    args.tft_fft_mode_select = canonical
    return canonical


def _normalize_extension_mode(value, *, field):
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be one of {list(TFT_EXTENSION_INTEGRATION_MODES)}, "
            f"got {type(value).__name__}."
        )
    mode = value.strip().lower()
    if mode not in TFT_EXTENSION_INTEGRATION_MODES:
        raise ValueError(
            f"{field} must be one of {list(TFT_EXTENSION_INTEGRATION_MODES)}, "
            f"got {value!r}."
        )
    return mode


def _capture_extension_mode_requests(args):
    """Retain explicit requests while allowing paired references to flip flags.

    ``apply_tft_profile`` may run more than once, and SR01 creates a reference
    by copying resolved variant args and disabling an extension flag.  Resolved
    mode attributes therefore cannot themselves be mistaken for new explicit
    requests on the second pass.
    """

    prior_requests = getattr(args, "_tft_extension_mode_requests", None)
    prior_resolved = getattr(args, "tft_resolved_extension_modes", None)
    if not isinstance(prior_requests, dict):
        prior_requests = {}

    requests = {}
    for name, spec in TFT_EXTENSION_INTEGRATION_SPECS.items():
        mode_key = spec["mode_key"]
        current = _normalize_extension_mode(
            getattr(args, mode_key, None), field=mode_key
        )
        previous_resolved = (
            prior_resolved.get(name)
            if isinstance(prior_resolved, dict)
            else None
        )
        if name in prior_requests and current == previous_resolved:
            requested = prior_requests[name]
        else:
            requested = current
        requests[name] = requested

    args._tft_extension_mode_requests = dict(requests)
    return requests


def _resolve_residual_and_position_contract(args):
    residual_shape = str(
        getattr(args, "tft_extension_residual_shape", "scalar")
    ).strip().lower()
    if residual_shape not in TFT_EXTENSION_RESIDUAL_SHAPES:
        raise ValueError(
            "tft_extension_residual_shape must be one of "
            f"{list(TFT_EXTENSION_RESIDUAL_SHAPES)}, got {residual_shape!r}."
        )
    args.tft_extension_residual_shape = residual_shape

    small_init_raw = getattr(args, "tft_small_residual_init", 1e-3)
    if isinstance(small_init_raw, bool):
        raise ValueError("tft_small_residual_init must be a positive finite number.")
    try:
        small_init = float(small_init_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "tft_small_residual_init must be a positive finite number."
        ) from exc
    if not math.isfinite(small_init) or small_init <= 0.0:
        raise ValueError("tft_small_residual_init must be a positive finite number.")
    args.tft_small_residual_init = small_init

    position_unit = str(getattr(args, "tft_position_unit", "steps")).strip().lower()
    if position_unit not in TFT_POSITION_UNITS:
        raise ValueError(
            f"tft_position_unit must be one of {list(TFT_POSITION_UNITS)}, "
            f"got {position_unit!r}."
        )
    position_source = str(
        getattr(args, "tft_position_source", "row_index")
    ).strip().lower()
    if position_source not in TFT_POSITION_SOURCES:
        raise ValueError(
            f"tft_position_source must be one of {list(TFT_POSITION_SOURCES)}, "
            f"got {position_source!r}."
        )
    feature_name = getattr(args, "tft_position_feature_name", None)
    if feature_name is not None:
        feature_name = str(feature_name).strip() or None
    if position_source == "known_feature" and feature_name is None:
        raise ValueError(
            "tft_position_feature_name is required when "
            "tft_position_source='known_feature'."
        )
    if position_source != "known_feature" and feature_name is not None:
        raise ValueError(
            "tft_position_feature_name is only valid when "
            "tft_position_source='known_feature'."
        )
    if position_source == "row_index" and position_unit != "steps":
        raise ValueError(
            "tft_position_source='row_index' has unit 'steps'; physical "
            "trading-session or calendar-day coordinates require "
            "explicit_argument or known_feature."
        )
    declared_regular_sampling = getattr(
        args, "tft_declared_regular_sampling", False
    )
    if not isinstance(declared_regular_sampling, bool):
        raise ValueError("tft_declared_regular_sampling must be a boolean.")
    semantics_version = _normalize_semantics_version(
        getattr(
            args,
            "tft_extension_semantics_version",
            TFT_CURRENT_EXTENSION_SEMANTICS_VERSION,
        )
    )
    declaration_source = getattr(
        args, "tft_regular_sampling_declaration_source", None
    )
    if declaration_source == "built_in_dataset_contract" and not (
        position_source == "row_index"
        and str(getattr(args, "data", "")) in TFT_REGULAR_SAMPLING_DATASETS
    ):
        declared_regular_sampling = False
        declaration_source = None
    if declared_regular_sampling and declaration_source is None:
        declaration_source = "user"
    elif (
        position_source == "row_index"
        and str(getattr(args, "data", "")) in TFT_REGULAR_SAMPLING_DATASETS
    ):
        declared_regular_sampling = True
        declaration_source = "built_in_dataset_contract"
    if (
        semantics_version == TFT_CURRENT_EXTENSION_SEMANTICS_VERSION
        and position_source == "row_index"
        and declared_regular_sampling is not True
    ):
        raise ValueError(
            "Semantics-v2 row-index coordinates require the explicit policy "
            "tft_declared_regular_sampling=True. Use explicit_argument with "
            "calendar-day/session coordinates for irregular observations."
        )
    if (
        semantics_version == TFT_CURRENT_EXTENSION_SEMANTICS_VERSION
        and position_source != "row_index"
        and declared_regular_sampling
    ):
        raise ValueError(
            "tft_declared_regular_sampling is only valid when "
            "tft_position_source='row_index'."
        )
    args.tft_position_unit = position_unit
    args.tft_position_source = position_source
    args.tft_position_feature_name = feature_name
    args.tft_declared_regular_sampling = declared_regular_sampling
    args.tft_regular_sampling_declaration_source = declaration_source


def resolve_tft_extension_modes(args):
    """Resolve every additive/structural extension to one unambiguous mode.

    The existing boolean flag remains the activation authority for backward
    compatibility.  An explicit mode refines an enabled semantics-v2 branch;
    a disabled flag always resolves to ``off``.  Version 1 can only replay its
    historical path and version 2 can never label that path as repaired.
    """

    semantics_version = _normalize_semantics_version(
        getattr(
            args,
            "tft_extension_semantics_version",
            TFT_CURRENT_EXTENSION_SEMANTICS_VERSION,
        )
    )
    requests = _capture_extension_mode_requests(args)
    resolved = {}
    for name, spec in TFT_EXTENSION_INTEGRATION_SPECS.items():
        enabled = bool(getattr(args, spec["config_flag"], False))
        requested = requests[name]
        mode_key = spec["mode_key"]

        if semantics_version == TFT_LEGACY_EXTENSION_SEMANTICS_VERSION:
            if not enabled:
                mode = "off"
            else:
                if requested not in (None, "legacy"):
                    raise ValueError(
                        f"{mode_key}={requested!r} is incompatible with TFT "
                        "extension semantics version 1; an enabled v1 branch "
                        "must use legacy mode."
                    )
                mode = "legacy"
        else:
            if requested == "legacy":
                raise ValueError(
                    f"{mode_key}='legacy' is checkpoint-compatibility-only and "
                    "cannot be used with TFT extension semantics version 2."
                )
            if not enabled:
                mode = "off"
            else:
                if requested == "off":
                    raise ValueError(
                        f"{mode_key}='off' contradicts enabled flag "
                        f"{spec['config_flag']}=True. Disable the flag instead."
                    )
                mode = "neutral" if requested is None else requested

        setattr(args, mode_key, mode)
        resolved[name] = mode

    _resolve_residual_and_position_contract(args)
    args.tft_resolved_extension_modes = dict(resolved)
    return dict(resolved)


def get_tft_extension_mode(args, extension_name):
    """Return one resolved mode by migration-capability name."""

    if extension_name not in TFT_EXTENSION_INTEGRATION_SPECS:
        raise KeyError(
            f"Unknown TFT extension {extension_name!r}; expected one of "
            f"{sorted(TFT_EXTENSION_INTEGRATION_SPECS)}."
        )
    return resolve_tft_extension_modes(args)[extension_name]


def build_tft_extension_contract(args):
    """Return the deterministic residual/coordinate contract for metadata."""

    modes = resolve_tft_extension_modes(args)
    uses_small_residual_initialization = any(
        mode == "small_residual" for mode in modes.values()
    )
    return {
        "schema_version": 1,
        "mode_semantics": "initialization_policy",
        "resolved_modes": modes,
        "residual_shape": args.tft_extension_residual_shape,
        # An unused exploratory initializer is not part of model identity or
        # checkpoint compatibility.  Record it only when at least one branch
        # actually initializes in small_residual mode.
        "small_residual_init": (
            args.tft_small_residual_init
            if uses_small_residual_initialization
            else None
        ),
        "temporal_coordinates": {
            "position_unit": args.tft_position_unit,
            "position_source": args.tft_position_source,
            "position_feature_name": args.tft_position_feature_name,
            "declared_regular_sampling": args.tft_declared_regular_sampling,
            "regular_sampling_declaration_source": (
                args.tft_regular_sampling_declaration_source
            ),
            "valid_position_shapes": ["T", "B,T"],
            "valid_mask_semantics": "true_is_valid",
            "monotonicity": "strict_over_valid_positions",
        },
    }


def _active_tft_extensions(args):
    return {
        name: bool(getattr(args, capability["config_flag"], False))
        for name, capability in TFT_EXTENSION_MIGRATION_CAPABILITIES.items()
        if capability["config_flag"] is not None
    }


def _resolved_schema_payload(args):
    from utils.tft_schema import resolve_tft_schema

    schema = getattr(args, "_tft_resolved_schema", None)
    if schema is None:
        schema = resolve_tft_schema(args)
    return {
        "feature_names": list(schema.feature_names),
        "observed_positions": list(schema.observed_positions),
        "static_positions": list(schema.static_positions),
        "target_positions": list(schema.target_positions),
        "known_feature_names": list(schema.known_feature_names),
        "features_mode": schema.features_mode,
        "enc_in": schema.enc_in,
        "c_out": schema.c_out,
    }


def _active_contract_payload(args):
    active = _active_tft_extensions(args)
    extension_contract = build_tft_extension_contract(args)
    capabilities = {"core": TFT_EXTENSION_MIGRATION_CAPABILITIES["core"]}
    capabilities.update(
        {
            name: TFT_EXTENSION_MIGRATION_CAPABILITIES[name]
            for name, enabled in active.items()
            if enabled
        }
    )
    return {
        "extension_semantics_version": _normalize_semantics_version(
            getattr(
                args,
                "tft_extension_semantics_version",
                TFT_CURRENT_EXTENSION_SEMANTICS_VERSION,
            )
        ),
        "capabilities": capabilities,
        "extension_integration": extension_contract,
    }


def _payload_digest(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def pending_v2_artifact_extensions(args):
    if (
        _normalize_semantics_version(
            getattr(
                args,
                "tft_extension_semantics_version",
                TFT_CURRENT_EXTENSION_SEMANTICS_VERSION,
            )
        )
        != TFT_CURRENT_EXTENSION_SEMANTICS_VERSION
    ):
        return []
    active = _active_tft_extensions(args)
    return sorted(
        name
        for name, enabled in active.items()
        if enabled
        and TFT_EXTENSION_MIGRATION_CAPABILITIES[name]["v2_artifact_status"]
        != "released"
    )


def validate_tft_v2_artifact_readiness(args):
    pending = pending_v2_artifact_extensions(args)
    if pending:
        tasks = sorted(
            {
                TFT_EXTENSION_MIGRATION_CAPABILITIES[name]["repair_task"]
                for name in pending
            }
        )
        raise RuntimeError(
            "Refusing to stamp unrepaired legacy computations as TFT semantics v2. "
            f"Pending extensions: {', '.join(pending)}; required tasks: "
            f"{', '.join(tasks)}. Use semantics v1 only for explicit legacy replay."
        )
    return args


def _file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _checkpoint_residual_strength_summary(checkpoint_path):
    """Describe learned SR02 adapter strengths in a saved bare state dict."""

    state_dict = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state_dict, dict):
        raise TypeError(
            "TFT checkpoint metadata requires a bare state_dict mapping; "
            f"got {type(state_dict).__name__}."
        )

    strengths = {}
    for raw_name, value in sorted(state_dict.items()):
        name = raw_name[len("module."):] if raw_name.startswith("module.") else raw_name
        if not name.endswith("residual_adapter.residual_strength"):
            continue
        if not torch.is_tensor(value):
            raise TypeError(
                f"Residual strength {raw_name!r} must be a tensor, got "
                f"{type(value).__name__}."
            )
        values = value.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
        if not torch.isfinite(values).all():
            raise ValueError(
                f"Residual strength {raw_name!r} contains a non-finite value."
            )
        strengths[name] = values.tolist()

    flat = [abs(number) for values in strengths.values() for number in values]
    return {
        "schema_version": 1,
        "mode_semantics": "initialization_policy",
        "parameter_values": strengths,
        "tensor_count": len(strengths),
        "scalar_count": len(flat),
        "nonzero_scalar_count": sum(number != 0.0 for number in flat),
        "max_abs_strength": max(flat, default=0.0),
    }


def build_tft_semantics_manifest(
    args,
    artifact_kind="checkpoint",
    setting=None,
    checkpoint_path=None,
):
    """Build deterministic TFT semantic/checkpoint metadata.

    Runtime timestamps deliberately do not belong here: identical resolved
    configurations must produce identical semantic manifests.
    """

    capability_payload = json.dumps(
        TFT_EXTENSION_MIGRATION_CAPABILITIES,
        sort_keys=True,
        separators=(",", ":"),
    )
    active_contract = _active_contract_payload(args)
    extension_contract = build_tft_extension_contract(args)
    payload = {
        "manifest_schema_version": 1,
        "artifact_kind": str(artifact_kind),
        "model": getattr(args, "model", None),
        "extension_semantics_version": _normalize_semantics_version(
            getattr(
                args,
                "tft_extension_semantics_version",
                TFT_CURRENT_EXTENSION_SEMANTICS_VERSION,
            )
        ),
        "profile": getattr(args, "tft_profile", None),
        "config_digest": getattr(args, "tft_config_digest", None),
        "digest_schema": getattr(args, "tft_digest_schema", None),
        "setting": setting,
        "resolved_schema": _resolved_schema_payload(args),
        "active_extensions": _active_tft_extensions(args),
        "extension_integration": extension_contract,
        "migration_capability_hash": hashlib.sha256(
            capability_payload.encode("utf-8")
        ).hexdigest()[:12],
        "active_contract_hash": _payload_digest(active_contract)[:12],
        "migration_capabilities": deepcopy(TFT_EXTENSION_MIGRATION_CAPABILITIES),
    }
    if checkpoint_path is not None:
        payload["checkpoint"] = _file_record(checkpoint_path)
        payload["checkpoint_extension_state"] = (
            _checkpoint_residual_strength_summary(checkpoint_path)
        )
    return payload


def write_tft_semantics_metadata(
    directory,
    args,
    artifact_kind="checkpoint",
    filename=TFT_CHECKPOINT_METADATA_FILENAME,
    setting=None,
    checkpoint_path=None,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if artifact_kind in {"checkpoint", "result"}:
        validate_tft_v2_artifact_readiness(args)
    if artifact_kind == "checkpoint":
        checkpoint_path = (
            directory / "checkpoint.pth"
            if checkpoint_path is None
            else Path(checkpoint_path)
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Cannot write TFT checkpoint metadata before the checkpoint exists: "
                f"{checkpoint_path}"
            )
    destination = directory / filename
    payload = build_tft_semantics_manifest(
        args,
        artifact_kind=artifact_kind,
        setting=setting,
        checkpoint_path=checkpoint_path if artifact_kind == "checkpoint" else None,
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=f".{filename}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def read_tft_semantics_metadata(
    checkpoint_path,
    filename=TFT_CHECKPOINT_METADATA_FILENAME,
):
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.parent / filename
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"TFT semantics metadata must be a JSON object: {metadata_path}")
    if "extension_semantics_version" not in payload:
        raise ValueError(
            f"TFT semantics metadata lacks extension_semantics_version: {metadata_path}"
        )
    payload["extension_semantics_version"] = _normalize_semantics_version(
        payload["extension_semantics_version"]
    )
    return payload


def _sensitive_active_extensions(args, metadata=None):
    active = _active_tft_extensions(args)
    if metadata is not None:
        for name, enabled in metadata.get("active_extensions", {}).items():
            if name in active:
                active[name] = active[name] or bool(enabled)
    return sorted(
        name
        for name, enabled in active.items()
        if enabled
        and TFT_EXTENSION_MIGRATION_CAPABILITIES[name]["v1_to_v2"] == "retrain"
    )


def _validate_same_version_metadata(
    args,
    checkpoint_path,
    metadata,
    *,
    expected_setting=None,
):
    if metadata is None:
        return
    runtime = build_tft_semantics_manifest(
        args,
        artifact_kind="checkpoint",
        setting=expected_setting,
    )
    fields = (
        "model",
        "extension_semantics_version",
        "profile",
        "config_digest",
        "digest_schema",
        "resolved_schema",
        "active_extensions",
        "extension_integration",
        "active_contract_hash",
    )
    mismatches = [
        field
        for field in fields
        if metadata.get(field) != runtime.get(field)
    ]
    if expected_setting is not None and metadata.get("setting") != expected_setting:
        mismatches.append("setting")
    if mismatches:
        raise RuntimeError(
            "TFT checkpoint metadata does not match the resolved runtime "
            f"configuration: {', '.join(sorted(set(mismatches)))}."
        )

    checkpoint_record = metadata.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise RuntimeError(
            "Versioned TFT checkpoint metadata is not bound to checkpoint.pth; "
            "recreate the artifact under the matching runtime."
        )
    actual = _file_record(checkpoint_path)
    if checkpoint_record != actual:
        raise RuntimeError(
            "TFT checkpoint hash/size does not match checkpoint_metadata.json; "
            "the artifact is incomplete, moved incorrectly, or has been modified."
        )


def validate_tft_checkpoint_compatibility(
    args,
    checkpoint_path,
    external_load=True,
    expected_setting=None,
):
    """Resolve an explicit, non-silent TFT checkpoint load policy."""

    requested = _normalize_semantics_version(
        getattr(
            args,
            "tft_extension_semantics_version",
            TFT_CURRENT_EXTENSION_SEMANTICS_VERSION,
        )
    )
    metadata = read_tft_semantics_metadata(checkpoint_path)
    checkpoint_version = (
        TFT_LEGACY_EXTENSION_SEMANTICS_VERSION
        if metadata is None
        else metadata["extension_semantics_version"]
    )
    sensitive = _sensitive_active_extensions(args, metadata)
    allow_legacy = bool(
        getattr(args, "tft_allow_legacy_extension_checkpoint", False)
    )

    if checkpoint_version == requested == TFT_CURRENT_EXTENSION_SEMANTICS_VERSION:
        validate_tft_v2_artifact_readiness(args)
        _validate_same_version_metadata(
            args,
            checkpoint_path,
            metadata,
            expected_setting=expected_setting,
        )
        load_mode = "current"
    elif checkpoint_version == requested == TFT_LEGACY_EXTENSION_SEMANTICS_VERSION:
        if external_load and not allow_legacy:
            raise RuntimeError(
                "External legacy TFT checkpoint replay requires "
                "--tft_allow_legacy_extension_checkpoint and "
                "--tft_extension_semantics_version 1."
            )
        _validate_same_version_metadata(
            args,
            checkpoint_path,
            metadata,
            expected_setting=expected_setting,
        )
        load_mode = (
            "explicit_legacy_replay"
            if external_load
            else "trusted_current_run_legacy_reload"
        )
    elif (
        checkpoint_version == TFT_LEGACY_EXTENSION_SEMANTICS_VERSION
        and requested == TFT_CURRENT_EXTENSION_SEMANTICS_VERSION
    ):
        suffix = (
            " Affected extensions: " + ", ".join(sensitive)
            if sensitive
            else ""
        )
        raise RuntimeError(
            "Legacy TFT checkpoint weights cannot be mapped into semantics v2; "
            "retrain under v2, or explicitly replay the legacy graph with "
            "--tft_extension_semantics_version 1 "
            "--tft_allow_legacy_extension_checkpoint."
            + suffix
        )
    else:
        raise RuntimeError(
            f"TFT checkpoint semantics version {checkpoint_version} is incompatible "
            f"with requested runtime version {requested}."
        )

    return {
        "checkpoint_semantics_version": checkpoint_version,
        "runtime_semantics_version": requested,
        "load_mode": load_mode,
        "sensitive_extensions": sensitive,
        "metadata_present": metadata is not None,
    }


def compute_tft_config_digest(args):
    model_name = getattr(args, "model", None)
    semantics_version = _normalize_semantics_version(
        getattr(
            args,
            "tft_extension_semantics_version",
            TFT_CURRENT_EXTENSION_SEMANTICS_VERSION,
        )
    )
    resolved_extension_modes = resolve_tft_extension_modes(args)
    uses_small_residual_initialization = any(
        mode == "small_residual" for mode in resolved_extension_modes.values()
    )
    material = {
        "model": model_name,
        "task_name": getattr(args, "task_name", None),
        "enc_in": getattr(args, "enc_in", None),
        "c_out": getattr(args, "c_out", None),
        "d_model": getattr(args, "d_model", None),
        "n_heads": getattr(args, "n_heads", None),
        "e_layers": getattr(args, "e_layers", None),
        "d_ff": None if model_name == "TemporalFusionTransformer" else getattr(args, "d_ff", None),
        "dropout": getattr(args, "dropout", None),
        "embed": getattr(args, "embed", None),
        "freq": getattr(args, "freq", None),
        "features": getattr(args, "features", None),
        "tft_profile": getattr(args, "tft_profile", None),
        "tft_temporal_backbone_layers_scope": getattr(args, "tft_temporal_backbone_layers_scope", None),
    }
    digest_keys = (
        TFT_LEGACY_V1_DIGEST_KEYS
        if semantics_version == TFT_LEGACY_EXTENSION_SEMANTICS_VERSION
        else tuple(sorted(TFT_BASE_DEFAULTS))
    )
    for key in digest_keys:
        material[key] = getattr(args, key, None)
    if semantics_version == TFT_CURRENT_EXTENSION_SEMANTICS_VERSION:
        # Version 2 makes the effective, fully resolved schema part of model
        # identity. Version 1 must preserve its historical digest exactly.
        if getattr(args, "task_name", None) != "long_term_forecast":
            # Construction rejects unsupported tasks before schema lookup; do
            # not let digesting an M4/short-term request mask that precise error.
            material["resolved_schema"] = "unsupported_task"
        else:
            material["resolved_schema"] = _resolved_schema_payload(args)
        for key in sorted(TFT_V2_ONLY_DEFAULTS):
            value = getattr(args, key, None)
            if key == "tft_small_residual_init" and not uses_small_residual_initialization:
                value = None
            material[key] = value
        extension_contract = build_tft_extension_contract(args)
        if not uses_small_residual_initialization:
            extension_contract = deepcopy(extension_contract)
            extension_contract["small_residual_init"] = None
        material["extension_integration"] = extension_contract
    payload = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _normalize_native_tft_effective_config(args):
    if getattr(args, "model", None) != "TemporalFusionTransformer":
        return

    ignored_knobs = []
    for key, default_value in TFT_IGNORED_MODEL_KNOBS.items():
        current = getattr(args, key, None)
        if current != default_value:
            warnings.warn(
                f"TemporalFusionTransformer ignores {key}; got {current!r} and will resolve it as non-material.",
                UserWarning,
                stacklevel=3,
            )
            ignored_knobs.append(key)
    args.tft_ignored_model_knobs = tuple(sorted(set(ignored_knobs)))
    args.tft_temporal_backbone_layers_scope = "gated_tcn_and_hybrid_tcn_lstm_only"


def apply_tft_profile(args):
    profile = _normalize_profile_name(getattr(args, "tft_profile", "extended_safe"))
    setattr(args, "tft_profile", profile)

    for key, value in TFT_BASE_DEFAULTS.items():
        _setdefault_attr(args, key, value)

    args.tft_extension_semantics_version = _normalize_semantics_version(
        args.tft_extension_semantics_version
    )
    _normalize_fft_mode_select(args)

    for key, value in TFT_PROFILE_DEFAULTS[profile].items():
        _maybe_apply_profile_override(args, key, value)

    _normalize_fft_mode_select(args)

    for key, value in TFT_V2_ONLY_DEFAULTS.items():
        _setdefault_attr(args, key, value)

    if getattr(args, "tft_output_mode", None) in (None, ""):
        args.tft_output_mode = "joint" if bool(getattr(args, "tft_use_quantile_head", False)) else "point"

    resolve_tft_extension_modes(args)
    validate_tft_profile(args)
    _normalize_native_tft_effective_config(args)
    args.tft_config_digest = compute_tft_config_digest(args)
    args.tft_digest_schema = (
        "legacy-v1"
        if args.tft_extension_semantics_version
        == TFT_LEGACY_EXTENSION_SEMANTICS_VERSION
        else "semantics-v2"
    )
    return args


def validate_tft_profile(args):
    profile = _normalize_profile_name(getattr(args, "tft_profile", "extended_safe"))
    _normalize_semantics_version(
        getattr(
            args,
            "tft_extension_semantics_version",
            TFT_CURRENT_EXTENSION_SEMANTICS_VERSION,
        )
    )

    if profile == "canonical":
        mismatches = []
        for key, expected in TFT_CANONICAL_REQUIRED.items():
            actual = getattr(args, key, None)
            if actual != expected:
                mismatches.append(f"{key}={actual!r} (expected {expected!r})")
        if mismatches:
            raise ValueError(
                "tft_profile='canonical' rejects incompatible overrides: " + ", ".join(mismatches)
            )

    output_mode = getattr(args, "tft_output_mode", None)
    use_quantile_head = bool(getattr(args, "tft_use_quantile_head", False))
    if output_mode in {"quantile", "joint"} and not use_quantile_head:
        raise ValueError(
            f"tft_profile={profile!r} resolved tft_output_mode={output_mode!r} but tft_use_quantile_head=False."
        )

    if getattr(args, "tft_temporal_backbone", None) not in {"lstm", "gated_tcn", "hybrid_tcn_lstm"}:
        raise ValueError("tft_temporal_backbone must be one of: lstm, gated_tcn, hybrid_tcn_lstm.")

    return args
