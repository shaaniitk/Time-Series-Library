from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from layers.TemporalFusion_layers import MultiScaleLagAttention
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
        "model_id": "tft-sr05-lag-contract",
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
        "tft_use_lag_attention": True,
        "tft_lag_scales": [2],
        "tft_lag_semantics_mode": "shifted_prefix_attention",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolved_config(**overrides):
    return apply_tft_profile(_raw_config(**overrides))


def _inputs():
    generator = torch.Generator().manual_seed(15071993)
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


def test_exact_token_lag_recovers_only_declared_impulse_source():
    sequence = torch.zeros(1, 8, 1)
    sequence[0, 3, 0] = 5.0
    attention = MultiScaleLagAttention(
        1,
        1,
        lag_scales=[2],
        lag_semantics_mode="exact_token_lag",
        dropout=0.0,
        extension_semantics_version=2,
    ).eval()

    with torch.no_grad():
        output, payload = attention(sequence, return_attention=True)

    # Only token t=5 should receive the lag-2 impulse from source t=3.
    shifted = torch.zeros_like(sequence)
    shifted[0, 5, 0] = 5.0
    expected = attention.out_projection(shifted)
    torch.testing.assert_close(output, expected)
    assert payload["lag_semantics_mode"] == "exact_token_lag"
    assert payload["lag_attention_mode"] == "exact_token_lag"
    weights = payload["lag_attention"][0, 0, :, :, 0]
    assert weights[5, 3].item() == pytest.approx(1.0)
    # Queries at t<lag are invalid and keys in the padded prefix are masked,
    # so only the causal/key-valid exact-lag pairs remain active.
    assert torch.count_nonzero(weights).item() == 4


def test_shifted_prefix_and_exact_token_modes_are_distinct():
    torch.manual_seed(7)
    sequence = torch.randn(2, 8, 4)
    shifted = MultiScaleLagAttention(
        4,
        2,
        lag_scales=[2],
        lag_semantics_mode="shifted_prefix_attention",
        dropout=0.0,
        extension_semantics_version=2,
    ).eval()
    exact = MultiScaleLagAttention(
        4,
        2,
        lag_scales=[2],
        lag_semantics_mode="exact_token_lag",
        dropout=0.0,
        extension_semantics_version=2,
    ).eval()
    exact.load_state_dict(shifted.state_dict(), strict=True)

    with torch.no_grad():
        shifted_output, shifted_payload = shifted(sequence, return_attention=True)
        exact_output, exact_payload = exact(sequence, return_attention=True)

    assert shifted_payload["lag_semantics_mode"] == "shifted_prefix_attention"
    assert shifted_payload["lag_attention_mode"] == "shifted_history_attention"
    assert exact_payload["lag_semantics_mode"] == "exact_token_lag"
    assert not torch.equal(shifted_output, exact_output)


def test_elapsed_time_response_mode_is_explicitly_delegated():
    attention = MultiScaleLagAttention(
        4,
        2,
        lag_scales=[1],
        lag_semantics_mode="elapsed_time_response",
        dropout=0.0,
        extension_semantics_version=2,
    ).eval()

    with pytest.raises(NotImplementedError, match="calendar-time response-bank"):
        attention(torch.randn(1, 6, 4))


def test_v2_model_reports_selected_lag_semantics_mode():
    model = NativeTFT(
        _raw_config(tft_lag_semantics_mode="exact_token_lag")
    ).eval()

    with torch.no_grad():
        payload = model(*_inputs(), return_interpretation=True)

    assert payload["lag_semantics_mode"] == "exact_token_lag"
    assert payload["lag_attention_weights"].shape[-1] == 1


def test_lag_neutral_mode_retains_exact_reference_parity():
    torch.manual_seed(10)
    reference = NativeTFT(
        _raw_config(tft_use_lag_attention=False)
    ).eval()
    torch.manual_seed(11)
    variant = NativeTFT(
        _raw_config(tft_lag_semantics_mode="exact_token_lag")
    ).eval()
    _copy_shared_reference_state(reference, variant)

    with torch.no_grad():
        reference_output = reference(*_inputs())
        variant_output = variant(*_inputs())

    assert torch.equal(variant_output, reference_output)


def test_sr05_releases_lag_attention_v2_artifacts():
    args = _resolved_config()

    assert (
        TFT_EXTENSION_MIGRATION_CAPABILITIES["lag_attention"]["v2_artifact_status"]
        == "released"
    )
    assert pending_v2_artifact_extensions(args) == []
    assert validate_tft_v2_artifact_readiness(args) is args
