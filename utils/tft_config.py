import hashlib
import json
import warnings
from copy import deepcopy


TFT_BASE_DEFAULTS = {
    "tft_profile": "extended_safe",
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


def compute_tft_config_digest(args):
    model_name = getattr(args, "model", None)
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
    for key in sorted(TFT_BASE_DEFAULTS.keys()):
        material[key] = getattr(args, key, None)
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

    for key, value in TFT_PROFILE_DEFAULTS[profile].items():
        _maybe_apply_profile_override(args, key, value)

    if getattr(args, "tft_output_mode", None) in (None, ""):
        args.tft_output_mode = "joint" if bool(getattr(args, "tft_use_quantile_head", False)) else "point"

    validate_tft_profile(args)
    _normalize_native_tft_effective_config(args)
    args.tft_config_digest = compute_tft_config_digest(args)
    return args


def validate_tft_profile(args):
    profile = _normalize_profile_name(getattr(args, "tft_profile", "extended_safe"))

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
