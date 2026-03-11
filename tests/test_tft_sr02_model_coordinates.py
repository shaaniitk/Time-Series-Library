from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.TemporalFusionTransformer import Model as NativeTFT
from utils.tft_config import apply_tft_profile
from utils.tft_coordinates import TemporalCoordinateContract


def _config(**overrides):
    values = {
        "model": "TemporalFusionTransformer",
        "task_name": "long_term_forecast",
        "data": "sr02_coordinate_test",
        "features": "MS",
        "seq_len": 6,
        "label_len": 2,
        "pred_len": 3,
        "enc_in": 3,
        "dec_in": 3,
        "c_out": 1,
        "d_model": 8,
        "n_heads": 2,
        "e_layers": 1,
        "d_layers": 1,
        "dropout": 0.0,
        "embed": "timeF",
        "freq": "h",
        "tft_profile": "extended_safe",
        "tft_extension_semantics_version": 2,
        "tft_temporal_backbone": "lstm",
        "tft_full_attention": True,
        "tft_attention_backend": "exact",
        "tft_attention_position_bias": "none",
        "tft_allow_custom_known": True,
        "tft_known_len": 2,
        "tft_known_max_channels": 8,
        "tft_known_feature_names": ["elapsed", "known_signal"],
        "tft_observed_pos": [0, 1, 2],
        "tft_static_pos": [],
        "tft_target_pos": [2],
    }
    values.update(overrides)
    return apply_tft_profile(SimpleNamespace(**values))


def _inputs(*, batch_size: int = 2):
    generator = torch.Generator().manual_seed(20260801)
    x_enc = torch.randn(batch_size, 6, 3, generator=generator)
    x_mark_enc = torch.randn(batch_size, 6, 2, generator=generator)
    x_dec = torch.randn(batch_size, 5, 3, generator=generator)
    x_mark_dec = torch.randn(batch_size, 5, 2, generator=generator)
    return x_enc, x_mark_enc, x_dec, x_mark_dec


def _make_explicit_contract(
    positions,
    *,
    unit: str = "calendar_days",
    valid_mask=None,
    source: str = "explicit_argument",
    source_name: str | None = None,
):
    return TemporalCoordinateContract(
        positions,
        unit=unit,
        valid_mask=valid_mask,
        source=source,
        source_name=source_name,
    )


def _assert_exact(left: torch.Tensor, right: torch.Tensor) -> None:
    torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_v2_default_row_coordinates_are_explicit_bt_payload_contract():
    torch.manual_seed(1)
    model = NativeTFT(
        _config(tft_declared_regular_sampling=True)
    ).eval()
    inputs = _inputs(batch_size=2)

    with torch.no_grad():
        payload = model(*inputs, return_interpretation=True)

    expected = torch.arange(9, dtype=torch.float64).expand(2, -1)
    _assert_exact(payload["temporal_positions"].cpu(), expected)
    assert payload["temporal_positions"].shape == (2, 9)
    assert payload["temporal_valid_mask"].shape == (2, 9)
    assert bool(payload["temporal_valid_mask"].all())

    metadata = payload["temporal_coordinate_metadata"]
    assert metadata["shape"] == [2, 9]
    assert metadata["source"] == "row_index"
    assert metadata["unit"] == "steps"
    assert metadata["declared_regular_sampling"] is True
    assert metadata["true_means_valid"] is True


def test_explicit_shared_t_and_repeated_bt_coordinates_are_exactly_equivalent():
    torch.manual_seed(2)
    model = NativeTFT(
        _config(
            tft_position_source="explicit_argument",
            tft_position_unit="calendar_days",
            tft_attention_position_bias="rope",
        )
    ).eval()
    inputs = _inputs(batch_size=2)
    positions = torch.tensor(
        [0.0, 1.0, 4.0, 5.0, 9.0, 10.0, 14.0, 15.0, 21.0]
    )
    shared = _make_explicit_contract(positions)
    batched = _make_explicit_contract(positions.expand(2, -1).clone())

    with torch.no_grad():
        shared_payload = model(
            *inputs, return_interpretation=True, temporal_coordinates=shared
        )
        batched_payload = model(
            *inputs, return_interpretation=True, temporal_coordinates=batched
        )

    _assert_exact(shared_payload["predictions"], batched_payload["predictions"])
    _assert_exact(
        shared_payload["attention_weights_full"],
        batched_payload["attention_weights_full"],
    )
    _assert_exact(
        shared_payload["temporal_positions"], batched_payload["temporal_positions"]
    )
    assert shared_payload["temporal_positions"].shape == (2, 9)


@pytest.mark.parametrize("position_bias_type", ["rope", "alibi"])
def test_irregular_batched_coordinates_reach_native_rope_and_alibi(
    position_bias_type,
):
    torch.manual_seed(3)
    model = NativeTFT(
        _config(
            tft_position_source="explicit_argument",
            tft_position_unit="calendar_days",
            tft_attention_position_bias=position_bias_type,
        )
    ).eval()
    inputs = _inputs(batch_size=2)
    positions = torch.tensor(
        [
            [0.0, 1.0, 4.0, 5.0, 9.0, 10.0, 14.0, 15.0, 21.0],
            [0.0, 2.0, 3.0, 8.0, 11.0, 12.0, 13.0, 20.0, 25.0],
        ],
        dtype=torch.float64,
    )

    with torch.no_grad():
        payload = model(
            *inputs,
            return_interpretation=True,
            temporal_coordinates=_make_explicit_contract(positions),
        )

    _assert_exact(payload["temporal_positions"].cpu(), positions)
    assert payload["position_bias_type"] == position_bias_type
    assert torch.isfinite(payload["predictions"]).all()
    assert torch.isfinite(payload["attention_weights_full"]).all()
    assert payload["attention_weights_full"].shape == (2, 2, 9, 9)


@pytest.mark.parametrize(
    ("contract", "message"),
    [
        (
            lambda: _make_explicit_contract(torch.arange(8)),
            "length must equal history\\+future length",
        ),
        (
            lambda: _make_explicit_contract(
                torch.arange(9).expand(3, -1).clone()
            ),
            "batch size must be 1 or match",
        ),
        (
            lambda: _make_explicit_contract(torch.arange(9), unit="steps"),
            "unit .* does not match configured unit",
        ),
        (
            lambda: _make_explicit_contract(
                torch.arange(9),
                source="named_known_feature",
                source_name="elapsed",
            ),
            "source .* does not match configured source",
        ),
    ],
)
def test_model_rejects_wrong_coordinate_length_batch_unit_and_source(
    contract, message
):
    model = NativeTFT(
        _config(
            tft_position_source="explicit_argument",
            tft_position_unit="calendar_days",
        )
    ).eval()

    with pytest.raises(ValueError, match=message):
        model(*_inputs(batch_size=2), temporal_coordinates=contract())


def test_known_feature_coordinates_use_encoder_and_final_prediction_marks_only():
    torch.manual_seed(4)
    model = NativeTFT(
        _config(
            tft_position_source="known_feature",
            tft_position_unit="trading_sessions",
            tft_position_feature_name="elapsed",
            tft_attention_position_bias="alibi",
        )
    ).eval()
    x_enc, x_mark_enc, x_dec, x_mark_dec = _inputs(batch_size=2)
    encoder_positions = torch.tensor(
        [[0.0, 1.0, 2.0, 4.0, 5.0, 7.0], [10.0, 11.0, 13.0, 14.0, 17.0, 18.0]]
    )
    # The label-context rows are deliberately non-monotonic and out of range:
    # they are not decoder forecast tokens and must not enter the contract.
    decoder_positions = torch.tensor(
        [[-100.0, -50.0, 8.0, 10.0, 11.0], [-90.0, -40.0, 20.0, 23.0, 24.0]]
    )
    x_mark_enc[:, :, 0] = encoder_positions
    x_mark_dec[:, :, 0] = decoder_positions
    expected = torch.cat([encoder_positions, decoder_positions[:, -3:]], dim=1)

    with torch.no_grad():
        payload = model(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            return_interpretation=True,
        )

    _assert_exact(payload["temporal_positions"].cpu(), expected.to(torch.float64))
    metadata = payload["temporal_coordinate_metadata"]
    assert metadata["source"] == "named_known_feature"
    assert metadata["source_name"] == "elapsed"
    assert metadata["unit"] == "trading_sessions"


def test_true_valid_mask_removes_attention_keys_and_is_nan_safe():
    torch.manual_seed(5)
    model = NativeTFT(
        _config(
            tft_position_source="explicit_argument",
            tft_position_unit="calendar_days",
            tft_attention_position_bias="alibi",
        )
    ).eval()
    positions = torch.tensor(
        [
            [0.0, 1.0, 4.0, 5.0, 9.0, 10.0, 14.0, 15.0, 21.0],
            [0.0, 2.0, 3.0, 8.0, 11.0, 12.0, 13.0, 20.0, 25.0],
        ]
    )
    valid = torch.tensor(
        [
            [True, False, True, True, False, True, True, True, True],
            [True, True, False, True, True, False, True, True, True],
        ]
    )
    contract = _make_explicit_contract(positions, valid_mask=valid)

    with torch.no_grad():
        payload = model(
            *_inputs(batch_size=2),
            return_interpretation=True,
            temporal_coordinates=contract,
        )

    weights = payload["attention_weights_full"]
    assert torch.isfinite(payload["predictions"]).all()
    assert torch.isfinite(weights).all()
    _assert_exact(payload["temporal_valid_mask"].cpu(), valid)
    assert torch.count_nonzero(weights.masked_select(~valid[:, None, None, :])) == 0
    assert torch.count_nonzero(weights.masked_select(~valid[:, None, :, None])) == 0


@pytest.mark.parametrize(
    ("backbone", "use_revin"),
    [("lstm", False), ("gated_tcn", True), ("hybrid_tcn_lstm", True)],
)
def test_invalid_token_payload_is_exactly_invariant_through_normalization_and_branches(
    backbone,
    use_revin,
):
    torch.manual_seed(23)
    model = NativeTFT(
        _config(
            tft_position_source="explicit_argument",
            tft_position_unit="calendar_days",
            tft_temporal_backbone=backbone,
            tft_temporal_backbone_layers=1,
            tft_use_revin=use_revin,
            tft_revin_affine=True,
            tft_dual_attention_fusion=True,
            tft_use_regime_moe=True,
            tft_num_moe_experts=2,
            tft_num_regimes=2,
            tft_moe_top_k=1,
            tft_vsn_residual_bypass=True,
            tft_dual_attention_integration_mode="small_residual",
            tft_regime_moe_integration_mode="small_residual",
            tft_vsn_bypass_integration_mode="small_residual",
            tft_small_residual_init=0.1,
            tft_output_mode="joint",
            tft_use_quantile_head=True,
            tft_output_quantiles=[0.1, 0.5, 0.9],
        )
    ).eval()
    clean = list(_inputs(batch_size=2))
    valid = torch.tensor(
        [
            [True, False, True, True, False, True, True, False, True],
            [False, True, True, False, True, True, True, True, False],
        ],
        dtype=torch.bool,
    )
    positions = torch.tensor(
        [
            [0.0, 1.0, 3.0, 4.0, 7.0, 8.0, 10.0, 11.0, 14.0],
            [0.0, 2.0, 3.0, 5.0, 8.0, 9.0, 12.0, 13.0, 17.0],
        ],
        dtype=torch.float64,
    )
    contract = _make_explicit_contract(positions, valid_mask=valid)
    corrupted = [tensor.clone() for tensor in clean]
    history_invalid = ~valid[:, :6]
    future_invalid = ~valid[:, 6:]
    corrupted[0][history_invalid] = float("nan")
    corrupted[1][history_invalid] = float("inf")
    corrupted[2][:, -3:][future_invalid] = -1e30
    corrupted[3][:, -3:][future_invalid] = float("nan")

    with torch.no_grad():
        clean_output = model(
            *clean,
            return_auxiliary=True,
            temporal_coordinates=contract,
        )
        corrupted_output = model(
            *corrupted,
            return_auxiliary=True,
            temporal_coordinates=contract,
        )

    _assert_exact(clean_output.point_full, corrupted_output.point_full)
    _assert_exact(
        clean_output.quantile_forecast,
        corrupted_output.quantile_forecast,
    )
    assert torch.count_nonzero(
        clean_output.point_forecast.masked_select(future_invalid.unsqueeze(-1))
    ) == 0
    assert torch.count_nonzero(
        clean_output.quantile_forecast.masked_select(
            future_invalid.unsqueeze(-1).unsqueeze(-1)
        )
    ) == 0


def test_raw_coordinate_tensors_match_immutable_contract_exactly():
    torch.manual_seed(24)
    model = NativeTFT(
        _config(
            tft_position_source="explicit_argument",
            tft_position_unit="calendar_days",
        )
    ).eval()
    inputs = _inputs(batch_size=2)
    positions = torch.tensor(
        [0.0, 1.0, 4.0, 5.0, 9.0, 10.0, 14.0, 15.0, 21.0],
        dtype=torch.float64,
    ).expand(2, -1).clone()
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[0, 1] = False
    contract = _make_explicit_contract(positions, valid_mask=valid)

    with torch.no_grad():
        from_contract = model(*inputs, temporal_coordinates=contract)
        from_tensors = model(
            *inputs,
            temporal_positions=positions,
            temporal_valid_mask=valid,
        )

    _assert_exact(from_contract, from_tensors)


def test_v1_rejects_explicit_coordinates_and_preserves_default_legacy_path():
    torch.manual_seed(6)
    model = NativeTFT(
        _config(
            tft_extension_semantics_version=1,
            tft_attention_position_bias="rope",
        )
    ).eval()
    inputs = _inputs(batch_size=2)

    with torch.no_grad():
        before = model(*inputs, return_interpretation=True)

    explicit = _make_explicit_contract(
        torch.arange(9, dtype=torch.float64), unit="calendar_days"
    )
    with pytest.raises(
        ValueError, match="require TFT extension semantics version 2"
    ):
        model(*inputs, temporal_coordinates=explicit)

    with torch.no_grad():
        after = model(*inputs, return_interpretation=True)

    _assert_exact(before["predictions"], after["predictions"])
    _assert_exact(before["attention_weights_full"], after["attention_weights_full"])
    _assert_exact(before["temporal_positions"], torch.arange(9))
    assert before["temporal_positions"].ndim == 1
    assert before["temporal_valid_mask"] is None
    assert before["temporal_coordinate_metadata"] is None
