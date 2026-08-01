from __future__ import annotations

from types import SimpleNamespace

import torch

from models.TemporalFusionTransformer import Model as NativeTFT
from utils.tft_config import (
    TFT_EXTENSION_MIGRATION_CAPABILITIES,
    apply_tft_profile,
    pending_v2_artifact_extensions,
    validate_tft_v2_artifact_readiness,
)


def _raw_config(**overrides):
    values = {
        "model": "TemporalFusionTransformer",
        "task_name": "long_term_forecast",
        "model_id": "tft-sr04-cross-attention-contract",
        "data": "ETTh1",
        "features": "MS",
        "seq_len": 8,
        "label_len": 4,
        "pred_len": 3,
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 1,
        "d_model": 8,
        "n_heads": 2,
        "e_layers": 1,
        "d_layers": 1,
        "d_ff": 2048,
        "dropout": 0.0,
        "embed": "timeF",
        "freq": "h",
        "tft_profile": "extended_safe",
        "tft_extension_semantics_version": 2,
        "tft_temporal_backbone": "lstm",
        "tft_temporal_backbone_layers": 1,
        "tft_target_pos": [6],
        "tft_full_attention": False,
        "tft_attention_dropout": 0.0,
        "tft_use_explicit_cross_attention": True,
        "tft_cross_attention_type": "interpretable",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolved_config(**overrides):
    return apply_tft_profile(_raw_config(**overrides))


def _inputs():
    generator = torch.Generator().manual_seed(24071994)
    return (
        torch.randn(2, 8, 7, generator=generator),
        torch.randn(2, 8, 4, generator=generator),
        torch.randn(2, 7, 1, generator=generator),
        torch.randn(2, 7, 4, generator=generator),
    )


def _copy_shared_reference_state(reference, variant):
    reference_state = reference.state_dict()
    variant_state = variant.state_dict()
    missing_or_changed = {
        name
        for name, tensor in reference_state.items()
        if name not in variant_state
        or variant_state[name].shape != tensor.shape
        or variant_state[name].dtype != tensor.dtype
    }
    assert missing_or_changed == set()
    with torch.no_grad():
        for name, tensor in reference_state.items():
            variant_state[name].copy_(tensor)
    variant.load_state_dict(variant_state, strict=True)


def test_cross_attention_is_future_query_to_history_enrichment_only():
    torch.manual_seed(1)
    model = NativeTFT(_raw_config()).eval()
    layer = model.temporal_fusion_decoder.layers[0]
    calls = []
    original_forward = layer.cross_attention.forward

    def recording_forward(query, context, *args, **kwargs):
        calls.append((query.detach().clone(), context.detach().clone()))
        return original_forward(query, context, *args, **kwargs)

    layer.cross_attention.forward = recording_forward
    base_inputs = _inputs()
    perturbed_future_market = tuple(value.clone() for value in base_inputs)
    perturbed_future_market[2][:, -model.pred_len :, :] += 123.0
    perturbed_history = tuple(value.clone() for value in base_inputs)
    perturbed_history[0][:, :, -1] += 7.0

    with torch.no_grad():
        base_payload = model(*base_inputs, return_interpretation=True)
        base_calls = calls[:]
        calls.clear()
        future_payload = model(*perturbed_future_market, return_interpretation=True)
        future_calls = calls[:]
        calls.clear()
        history_payload = model(*perturbed_history, return_interpretation=True)
        history_calls = calls[:]

    assert len(base_calls) == len(future_calls) == len(history_calls) == 1
    base_query, base_context = base_calls[0]
    future_query, future_context = future_calls[0]
    history_query, history_context = history_calls[0]
    assert base_query.shape[1] == model.pred_len
    assert base_context.shape[1] == model.seq_len
    assert torch.equal(base_query, future_query)
    assert torch.equal(base_context, future_context)
    assert torch.equal(
        base_payload["cross_attention_weights"],
        future_payload["cross_attention_weights"],
    )
    assert not torch.equal(base_context, history_context)
    assert not torch.equal(
        base_payload["cross_attention_weights"],
        history_payload["cross_attention_weights"],
    )


def test_interpretable_cross_attention_reports_truthful_role_and_probabilities():
    model = NativeTFT(_raw_config()).eval()

    with torch.no_grad():
        payload = model(*_inputs(), return_interpretation=True)

    diagnostics = payload["cross_attention_diagnostics"]
    weights = payload["cross_attention_weights"]
    assert diagnostics["role"] == "future_query_to_history_enrichment"
    assert diagnostics["query_scope"] == "future"
    assert diagnostics["key_value_scope"] == "history"
    assert diagnostics["attention_type"] == "interpretable"
    assert diagnostics["interpretable"] is True
    assert diagnostics["attention_entropy_per_head"].shape == (model.n_heads,)
    assert torch.isfinite(diagnostics["attention_entropy_per_head"]).all()
    assert torch.isfinite(diagnostics["head_disagreement"])
    assert diagnostics["branch_knockout_delta"].item() == 0.0
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones_like(weights[..., 0]),
    )
    assert payload["interpretation_flags"]["uses_noninterpretable_attention_branch"] is False


def test_full_cross_attention_is_labeled_noninterpretable_without_changing_history_scope():
    model = NativeTFT(
        _raw_config(tft_cross_attention_type="full")
    ).eval()

    with torch.no_grad():
        payload = model(*_inputs(), return_interpretation=True)

    diagnostics = payload["cross_attention_diagnostics"]
    assert diagnostics["attention_type"] == "full"
    assert diagnostics["interpretable"] is False
    assert diagnostics["query_scope"] == "future"
    assert diagnostics["key_value_scope"] == "history"
    assert payload["interpretation_flags"]["uses_noninterpretable_attention_branch"] is True


def test_cross_attention_neutral_mode_retains_exact_reference_parity():
    torch.manual_seed(10)
    reference = NativeTFT(
        _raw_config(tft_use_explicit_cross_attention=False)
    ).eval()
    torch.manual_seed(11)
    variant = NativeTFT(_raw_config()).eval()
    _copy_shared_reference_state(reference, variant)

    with torch.no_grad():
        reference_output = reference(*_inputs())
        variant_output = variant(*_inputs())

    assert torch.equal(variant_output, reference_output)


def test_sr04_releases_explicit_cross_attention_v2_artifacts():
    args = _resolved_config()

    assert (
        TFT_EXTENSION_MIGRATION_CAPABILITIES["explicit_cross_attention"]["v2_artifact_status"]
        == "released"
    )
    assert pending_v2_artifact_extensions(args) == []
    assert validate_tft_v2_artifact_readiness(args) is args