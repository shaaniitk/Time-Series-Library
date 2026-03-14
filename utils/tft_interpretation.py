import json
from pathlib import Path

import torch


def _tensor_mean_list(tensor):
    return tensor.detach().float().cpu().reshape(-1).mean().item()


def _shape_list(tensor):
    return list(tensor.shape) if torch.is_tensor(tensor) else None


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


def summarize_tft_interpretation(payload, top_k=3):
    if not isinstance(payload, dict):
        raise TypeError(f"TFT interpretation payload must be a dict, got {type(payload)}.")

    history_weights = payload.get("history_vsn_weights")
    future_weights = payload.get("future_vsn_weights")
    lag_scale_weights = payload.get("lag_scale_weights")
    branch_weights = payload.get("attention_branch_weights")
    regime_probabilities = payload.get("regime_probabilities")
    expert_routing = payload.get("expert_routing")

    summary = {
        "prediction_shape": _shape_list(payload.get("predictions")),
        "prediction_full_shape": _shape_list(payload.get("predictions_full")),
        "decoder_num_layers": payload.get("decoder_num_layers"),
        "has_graph_attention": payload.get("history_graph_attention") is not None,
        "has_lag_attention": payload.get("lag_attention_weights") is not None,
        "has_higher_order": payload.get("interaction_contribution") is not None,
        "has_regime_moe": payload.get("expert_routing") is not None,
        "attention_branch_weights": branch_weights.detach().float().cpu().tolist() if torch.is_tensor(branch_weights) else None,
        "lag_scale_weights": lag_scale_weights.detach().float().cpu().tolist() if torch.is_tensor(lag_scale_weights) else None,
        "mean_regime_probabilities": regime_probabilities.detach().float().cpu().mean(dim=0).tolist() if torch.is_tensor(regime_probabilities) else None,
        "mean_expert_routing": expert_routing.detach().float().cpu().mean(dim=(0, 1)).tolist() if torch.is_tensor(expert_routing) else None,
        "moe_aux_loss": float(payload["moe_aux_loss"]) if payload.get("moe_aux_loss") is not None else None,
        "top_history_vsn_entries": _topk_from_weights(history_weights, top_k),
        "top_future_vsn_entries": _topk_from_weights(future_weights, top_k),
        "history_graph_attention_mean": _tensor_mean_list(payload["history_graph_attention"]) if torch.is_tensor(payload.get("history_graph_attention")) else None,
        "lag_attention_mean": _tensor_mean_list(payload["lag_attention_weights"]) if torch.is_tensor(payload.get("lag_attention_weights")) else None,
        "interaction_contribution_mean": _tensor_mean_list(payload["interaction_contribution"]) if torch.is_tensor(payload.get("interaction_contribution")) else None,
    }
    return summary


def export_tft_interpretation_summary(payload, output_path, top_k=3):
    summary = summarize_tft_interpretation(payload, top_k=top_k)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary