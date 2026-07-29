import json
from pathlib import Path

import torch


def _tensor_mean_list(tensor):
    return tensor.detach().float().cpu().reshape(-1).mean().item()


def _shape_list(tensor):
    return list(tensor.shape) if torch.is_tensor(tensor) else None


def _mean_over_non_feature_dims(tensor):
    if not torch.is_tensor(tensor):
        return None
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 0:
        return float(tensor.item())
    if tensor.ndim == 1:
        return tensor.tolist()
    reduce_dims = tuple(range(tensor.ndim - 1))
    return tensor.mean(dim=reduce_dims).tolist()


def _topk_from_weights(weights, top_k):
    if not torch.is_tensor(weights):
        return []
    flat = weights.detach().float().cpu().reshape(-1)
    if flat.numel() == 0:
        return []
    top_k = min(top_k, flat.numel())
    values, indices = torch.topk(flat, top_k)
    return [
        {
            "flat_index": int(index.item()),
            "value": float(value.item()),
        }
        for value, index in zip(values, indices)
    ]


def _topk_feature_weights(weights, feature_names, top_k):
    if not torch.is_tensor(weights):
        return []
    weights = weights.detach().float().cpu()
    if weights.ndim == 0:
        return []
    feature_dim = weights.shape[-1]
    if feature_dim == 0:
        return []
    reduce_dims = tuple(range(weights.ndim - 1))
    feature_scores = weights.mean(dim=reduce_dims) if reduce_dims else weights
    top_k = min(top_k, feature_scores.numel())
    values, indices = torch.topk(feature_scores.reshape(-1), top_k)
    feature_names = list(feature_names or [])
    entries = []
    for value, index in zip(values, indices):
        feature_index = int(index.item())
        feature_name = feature_names[feature_index] if feature_index < len(feature_names) else str(feature_index)
        entries.append(
            {
                "feature_index": feature_index,
                "feature_name": feature_name,
                "value": float(value.item()),
            }
        )
    return entries


def summarize_tft_interpretation(payload, top_k=3):
    if not isinstance(payload, dict):
        raise TypeError(f"TFT interpretation payload must be a dict, got {type(payload)}.")

    history_weights = payload.get("history_vsn_weights")
    future_weights = payload.get("future_vsn_weights")
    lag_scale_weights = payload.get("lag_scale_weights")
    branch_weights = payload.get("attention_branch_weights")
    regime_probabilities = payload.get("regime_probabilities")
    expert_routing = payload.get("expert_routing")
    quantile_predictions = payload.get("quantile_predictions")
    static_feature_names = payload.get("static_feature_names")
    static_weights = payload.get("static_vsn_weights")
    static_graph_attention = payload.get("static_graph_attention")
    history_feature_names = payload.get("history_feature_names")
    future_feature_names = payload.get("future_feature_names")
    observed_feature_names = payload.get("observed_feature_names")
    known_feature_names = payload.get("known_feature_names")
    interpretation_flags = payload.get("interpretation_flags") or {}

    summary = {
        "prediction_shape": _shape_list(payload.get("predictions")),
        "prediction_full_shape": _shape_list(payload.get("predictions_full")),
        "quantile_prediction_shape": _shape_list(quantile_predictions),
        "quantiles": payload.get("quantiles"),
        "decoder_num_layers": payload.get("decoder_num_layers"),
        "position_bias_type": payload.get("position_bias_type"),
        "temporal_backbone_type": payload.get("temporal_backbone_type"),
        "attention_backend_config": payload.get("attention_backend_config"),
        "attention_backend_used": payload.get("attention_backend_used"),
        "cross_attention_backend_used": payload.get("cross_attention_backend_used"),
        "fft_branch_active": payload.get("fft_gate_mean") is not None,
        "fft_gate_mean": payload.get("fft_gate_mean").detach().float().cpu().tolist() if torch.is_tensor(payload.get("fft_gate_mean")) else None,
        "tc_active": payload.get("tc_active", False),
        "tc_compressed_history_len": payload.get("tc_compressed_history_len"),
        "has_graph_attention": payload.get("history_graph_attention") is not None,
        "has_cross_attention": payload.get("cross_attention_weights") is not None,
        "has_lag_attention": payload.get("lag_attention_weights") is not None,
        "has_higher_order": payload.get("interaction_contribution") is not None,
        "has_regime_moe": payload.get("expert_routing") is not None,
        "has_quantile_predictions": quantile_predictions is not None,
        "observed_feature_names": list(observed_feature_names) if observed_feature_names is not None else None,
        "known_feature_names": list(known_feature_names) if known_feature_names is not None else None,
        "history_feature_names": list(history_feature_names) if history_feature_names is not None else None,
        "future_feature_names": list(future_feature_names) if future_feature_names is not None else None,
        "attention_branch_weights": branch_weights.detach().float().cpu().tolist() if torch.is_tensor(branch_weights) else None,
        "lag_scale_weights": lag_scale_weights.detach().float().cpu().tolist() if torch.is_tensor(lag_scale_weights) else None,
        "mean_regime_probabilities": _mean_over_non_feature_dims(regime_probabilities),
        "mean_expert_routing": expert_routing.detach().float().cpu().mean(dim=(0, 1)).tolist() if torch.is_tensor(expert_routing) else None,
        "moe_aux_loss": float(payload["moe_aux_loss"]) if payload.get("moe_aux_loss") is not None else None,
        "top_history_vsn_entries": _topk_feature_weights(history_weights, history_feature_names, top_k),
        "top_future_vsn_entries": _topk_feature_weights(future_weights, future_feature_names, top_k),
        "history_vsn_axis": "history_features",
        "future_vsn_axis": "future_known_features",
        "static_feature_names": list(static_feature_names) if static_feature_names is not None else None,
        "top_static_vsn_entries": {
            context_key: _topk_feature_weights(context_weights, static_feature_names, top_k)
            for context_key, context_weights in static_weights.items()
        } if isinstance(static_weights, dict) else None,
        "interpretation_flags": {
            "is_canonical_vsn_attribution": bool(interpretation_flags.get("is_canonical_vsn_attribution", False)),
            "uses_graph_pre_mixing": bool(interpretation_flags.get("uses_graph_pre_mixing", False)),
            "uses_vsn_bypass": bool(interpretation_flags.get("uses_vsn_bypass", False)),
            "uses_noninterpretable_attention_branch": bool(
                interpretation_flags.get("uses_noninterpretable_attention_branch", False)
            ),
            "uses_global_spectral_mixing": bool(interpretation_flags.get("uses_global_spectral_mixing", False)),
            "routing_is_detached": bool(interpretation_flags.get("routing_is_detached", False)),
        },
        "history_graph_attention_mean": _tensor_mean_list(payload["history_graph_attention"]) if torch.is_tensor(payload.get("history_graph_attention")) else None,
        "cross_attention_mean": _tensor_mean_list(payload["cross_attention_weights"]) if torch.is_tensor(payload.get("cross_attention_weights")) else None,
        "lag_attention_mean": _tensor_mean_list(payload["lag_attention_weights"]) if torch.is_tensor(payload.get("lag_attention_weights")) else None,
        "interaction_contribution_mean": _tensor_mean_list(payload["interaction_contribution"]) if torch.is_tensor(payload.get("interaction_contribution")) else None,
        "static_graph_attention_mean": {
            context_key: _tensor_mean_list(context_graph_attention) if torch.is_tensor(context_graph_attention) else None
            for context_key, context_graph_attention in static_graph_attention.items()
        } if isinstance(static_graph_attention, dict) else None,
    }
    return summary


def export_tft_interpretation_summary(payload, output_path, top_k=3):
    summary = summarize_tft_interpretation(payload, top_k=top_k)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
