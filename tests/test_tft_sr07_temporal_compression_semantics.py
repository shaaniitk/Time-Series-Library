from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from layers.TemporalFusion_layers import TemporalCompression
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
        "model_id": "tft-sr07-temporal-compression-contract",
        "data": "ETTh1",
        "features": "MS",
        "seq_len": 128,
        "label_len": 64,
        "pred_len": 16,
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
        "tft_attention_dropout": 0.0,
        "tft_full_attention": True,
        "tft_use_temporal_compression": True,
        "tft_temporal_compression_integration_mode": "small_residual",
        "tft_temporal_compression_mode": "kv_pool",
        "tft_tc_stride": 2,
        "tft_tc_threshold": 64,
        "tft_tc_min_long_sequence": 512,
        "tft_tc_experimental_short_window": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolved_config(**overrides):
    return apply_tft_profile(_raw_config(**overrides))


def _inputs():
    generator = torch.Generator().manual_seed(7052000)
    cfg = _raw_config()
    return (
        torch.randn(2, cfg.seq_len, 7, generator=generator),
        torch.randn(2, cfg.seq_len, 4, generator=generator),
        torch.randn(2, cfg.label_len + cfg.pred_len, 1, generator=generator),
        torch.randn(2, cfg.label_len + cfg.pred_len, 4, generator=generator),
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


def test_kv_pool_rejects_short_window_without_experimental_override():
    model = NativeTFT(
        _raw_config(tft_tc_experimental_short_window=False)
    ).eval()

    with pytest.raises(RuntimeError, match="short-window activation is disabled"):
        with torch.no_grad():
            model(*_inputs(), return_interpretation=True)


def test_kv_pool_reports_availability_coordinates_and_reduces_kv_length():
    model = NativeTFT(_raw_config()).eval()

    with torch.no_grad():
        payload = model(*_inputs(), return_interpretation=True)

    assert payload["tc_active"] is True
    assert payload["tc_mode"] == "kv_pool"
    kv_pool = payload["tc_kv_pooling"]
    assert kv_pool["history_pooled_len"] < kv_pool["history_original_len"]
    assert kv_pool["key_value_total_len"] < (_raw_config().seq_len + _raw_config().pred_len)
    assert kv_pool["availability_positions"].shape[1] == kv_pool["history_pooled_len"]
    assert kv_pool["content_center_positions"].shape == kv_pool["availability_positions"].shape


def test_temporal_compression_off_or_inactive_threshold_is_exact_parity():
    torch.manual_seed(11)
    reference = NativeTFT(_raw_config(tft_use_temporal_compression=False)).eval()
    torch.manual_seed(12)
    variant = NativeTFT(_raw_config(tft_tc_threshold=9999)).eval()
    _copy_shared_reference_state(reference, variant)

    with torch.no_grad():
        reference_output = reference(*_inputs())
        variant_output = variant(*_inputs())

    assert torch.equal(variant_output, reference_output)


def test_kv_pool_component_tracks_availability_and_validity_coordinates():
    module = TemporalCompression(
        d_model=2,
        stride=2,
        threshold=0,
        mode="kv_pool",
        min_long_sequence=512,
        experimental_short_window=True,
    ).eval()
    x = torch.randn(1, 5, 2)
    positions = torch.tensor([0.0, 1.0, 3.0, 6.0, 10.0])
    valid_mask = torch.tensor([[True, True, False, True, True]])

    with torch.no_grad():
        pooled, metadata = module.pool_history_kv(x, positions=positions, valid_mask=valid_mask)

    assert pooled.shape == (1, 3, 2)
    torch.testing.assert_close(
        metadata["availability_positions"][0],
        torch.tensor([1.0, 6.0, 10.0]),
    )
    torch.testing.assert_close(
        metadata["content_center_positions"][0],
        torch.tensor([0.5, 6.0, 10.0]),
    )
    assert metadata["pooled_valid_mask"].tolist() == [[True, True, True]]


def test_kv_pool_path_has_live_gradients_with_small_residual_mode():
    model = NativeTFT(
        _raw_config(tft_temporal_compression_integration_mode="small_residual")
    ).train()

    output = model(*_inputs())
    loss = output.square().mean()
    loss.backward()

    compression_params = {
        name: param
        for name, param in model.named_parameters()
        if "temporal_compression" in name and param.requires_grad
    }
    assert compression_params
    active_grads = {
        name: param.grad
        for name, param in compression_params.items()
        if param.grad is not None
    }
    assert active_grads
    assert any(grad.abs().sum().item() > 0.0 for grad in active_grads.values())
    assert all(torch.isfinite(grad).all() for grad in active_grads.values())


def test_sr07_releases_temporal_compression_v2_artifacts():
    args = _resolved_config()

    compression = TFT_EXTENSION_MIGRATION_CAPABILITIES["temporal_compression"]
    assert compression["repair_task"] == "TFT-SR07"
    assert compression["v2_artifact_status"] == "released"

    assert pending_v2_artifact_extensions(args) == []
    assert validate_tft_v2_artifact_readiness(args) is args
