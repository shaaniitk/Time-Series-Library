from copy import deepcopy
from types import SimpleNamespace

import pytest

from run import build_parser, normalize_args
from utils.tft_config import (
    TFT_EXTENSION_INTEGRATION_SPECS,
    TFT_EXTENSION_MIGRATION_CAPABILITIES,
    apply_tft_profile,
    build_tft_semantics_manifest,
    compute_tft_config_digest,
    get_tft_extension_mode,
    pending_v2_artifact_extensions,
    resolve_tft_extension_modes,
    validate_tft_v2_artifact_readiness,
)


def _args(**overrides):
    values = {
        "model": "TemporalFusionTransformer",
        "task_name": "long_term_forecast",
        "model_id": "sr02-config-test",
        "data": "ETTh1",
        "features": "MS",
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 24,
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 1,
        "d_model": 16,
        "n_heads": 2,
        "e_layers": 1,
        "d_layers": 1,
        "d_ff": 2048,
        "dropout": 0.1,
        "embed": "timeF",
        "freq": "h",
        "tft_profile": "extended_safe",
    }
    values.update(overrides)
    return apply_tft_profile(SimpleNamespace(**values))


@pytest.mark.parametrize(
    ("extension_name", "flag", "mode_key"),
    [
        (name, spec["config_flag"], spec["mode_key"])
        for name, spec in TFT_EXTENSION_INTEGRATION_SPECS.items()
    ],
)
def test_every_registered_extension_resolves_off_legacy_and_neutral(
    extension_name, flag, mode_key
):
    legacy_off = _args(tft_extension_semantics_version=1)
    legacy_on = _args(
        tft_extension_semantics_version=1,
        **{flag: True},
    )
    current_off = _args(tft_extension_semantics_version=2)
    current_on = _args(
        tft_extension_semantics_version=2,
        **{flag: True},
    )

    assert getattr(legacy_off, mode_key) == "off"
    assert getattr(legacy_on, mode_key) == "legacy"
    assert getattr(current_off, mode_key) == "off"
    assert getattr(current_on, mode_key) == "neutral"
    assert get_tft_extension_mode(current_on, extension_name) == "neutral"


def test_explicit_small_residual_is_resolved_only_for_enabled_v2_branch():
    enabled = _args(
        tft_use_fft_branch=True,
        tft_fft_integration_mode="small_residual",
    )
    disabled = _args(tft_fft_integration_mode="small_residual")

    assert enabled.tft_fft_integration_mode == "small_residual"
    assert disabled.tft_fft_integration_mode == "off"
    assert enabled.tft_small_residual_init == pytest.approx(1e-3)


@pytest.mark.parametrize("bad_mode", ["sometimes", 1, [], object()])
def test_invalid_integration_mode_is_rejected(bad_mode):
    with pytest.raises(ValueError, match="tft_fft_integration_mode"):
        _args(
            tft_use_fft_branch=True,
            tft_fft_integration_mode=bad_mode,
        )


def test_legacy_mode_is_v1_only_and_enabled_v1_cannot_request_v2_modes():
    with pytest.raises(ValueError, match="checkpoint-compatibility-only"):
        _args(tft_fft_integration_mode="legacy")
    with pytest.raises(ValueError, match="semantics version 1"):
        _args(
            tft_extension_semantics_version=1,
            tft_use_fft_branch=True,
            tft_fft_integration_mode="neutral",
        )
    with pytest.raises(ValueError, match="contradicts enabled flag"):
        _args(tft_use_fft_branch=True, tft_fft_integration_mode="off")


@pytest.mark.parametrize(
    "bad_value",
    [0.0, -1e-3, float("nan"), float("inf"), True, "not-a-number"],
)
def test_small_residual_initialization_must_be_positive_and_finite(bad_value):
    with pytest.raises(ValueError, match="positive finite"):
        _args(tft_small_residual_init=bad_value)


def test_residual_shape_and_physical_position_contract_are_validated():
    channel = _args(tft_extension_residual_shape="CHANNEL")
    calendar = _args(
        tft_position_source="explicit_argument",
        tft_position_unit="calendar_days",
    )
    known = _args(
        tft_position_source="known_feature",
        tft_position_unit="trading_sessions",
        tft_position_feature_name="elapsed_trading_sessions",
    )

    assert channel.tft_extension_residual_shape == "channel"
    assert calendar.tft_position_unit == "calendar_days"
    assert known.tft_position_feature_name == "elapsed_trading_sessions"

    with pytest.raises(ValueError, match="tft_extension_residual_shape"):
        _args(tft_extension_residual_shape="matrix")
    with pytest.raises(ValueError, match="unit 'steps'"):
        _args(tft_position_unit="calendar_days")
    with pytest.raises(ValueError, match="tft_position_feature_name is required"):
        _args(tft_position_source="known_feature")
    with pytest.raises(ValueError, match="only valid"):
        _args(tft_position_feature_name="elapsed_days")


def test_v1_digest_ignores_all_v2_only_residual_and_coordinate_fields():
    legacy = _args(tft_extension_semantics_version=1)
    frozen = legacy.tft_config_digest

    legacy.tft_extension_residual_shape = "channel"
    legacy.tft_small_residual_init = 0.25
    legacy.tft_position_source = "explicit_argument"
    legacy.tft_position_unit = "calendar_days"
    legacy.tft_fft_integration_mode = "small_residual"

    assert compute_tft_config_digest(legacy) == frozen


def test_known_legacy_matrix_digest_remains_exact():
    argv = (
        "--task_name long_term_forecast --is_training 1 "
        "--model_id tft_ot_p24_baseline --model TemporalFusionTransformer "
        "--data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv "
        "--features MS --target OT --freq h --seq_len 96 --label_len 48 "
        "--pred_len 24 --enc_in 7 --dec_in 7 --c_out 1 --tft_target_pos 6 "
        "--d_model 16 --n_heads 2 --e_layers 1 --d_layers 1 --dropout 0.2 "
        "--learning_rate 5e-5 --lradj cosine --train_epochs 30 "
        "--batch_size 64 --patience 6 --loss MSE --itr 1 "
        "--des tft_ot_p24_baseline --tft_profile extended_safe "
        "--tft_temporal_backbone lstm --tft_temporal_backbone_layers 1 "
        "--tft_temporal_kernel_size 3 --tft_attention_dropout 0.08 "
        "--tft_extension_semantics_version 1 --no_use_gpu"
    ).split()
    args = normalize_args(build_parser().parse_args(argv))

    assert args.tft_config_digest == "6c54053ec5d7"


def test_v2_digest_tracks_mode_residual_and_coordinate_semantics():
    neutral = _args(tft_use_fft_branch=True)
    small = _args(
        tft_use_fft_branch=True,
        tft_fft_integration_mode="small_residual",
    )
    channel = _args(
        tft_use_fft_branch=True,
        tft_extension_residual_shape="channel",
    )
    calendar = _args(
        tft_position_source="explicit_argument",
        tft_position_unit="calendar_days",
    )

    assert len(
        {
            neutral.tft_config_digest,
            small.tft_config_digest,
            channel.tft_config_digest,
            calendar.tft_config_digest,
        }
    ) == 4


def test_v2_digest_ignores_unused_small_residual_initialization():
    neutral_default = _args(tft_use_fft_branch=True)
    neutral_changed = _args(
        tft_use_fft_branch=True,
        tft_small_residual_init=0.25,
    )
    small_default = _args(
        tft_use_fft_branch=True,
        tft_fft_integration_mode="small_residual",
    )
    small_changed = _args(
        tft_use_fft_branch=True,
        tft_fft_integration_mode="small_residual",
        tft_small_residual_init=0.25,
    )

    assert neutral_default.tft_config_digest == neutral_changed.tft_config_digest
    assert (
        build_tft_semantics_manifest(neutral_changed)["extension_integration"][
            "small_residual_init"
        ]
        is None
    )
    assert small_default.tft_config_digest != small_changed.tft_config_digest


def test_manifest_records_resolved_modes_and_coordinate_contract():
    args = _args(
        tft_use_fft_branch=True,
        tft_fft_integration_mode="small_residual",
        tft_extension_residual_shape="channel",
        tft_small_residual_init=0.02,
        tft_position_source="explicit_argument",
        tft_position_unit="calendar_days",
    )
    manifest = build_tft_semantics_manifest(args)
    contract = manifest["extension_integration"]

    assert contract["resolved_modes"]["fft_branch"] == "small_residual"
    assert contract["residual_shape"] == "channel"
    assert contract["small_residual_init"] == pytest.approx(0.02)
    assert contract["temporal_coordinates"]["position_unit"] == "calendar_days"
    assert contract["temporal_coordinates"]["valid_mask_semantics"] == "true_is_valid"


def test_sr02_additive_repairs_are_released_while_compression_remains_pending():
    for name in ("regime_moe", "dual_attention_fusion"):
        capability = TFT_EXTENSION_MIGRATION_CAPABILITIES[name]
        assert capability["repair_task"] == "TFT-SR02"
        assert capability["v2_artifact_status"] == "released"
        assert capability["integration_kind"] == "additive"

    vsn_bypass = TFT_EXTENSION_MIGRATION_CAPABILITIES["vsn_residual_bypass"]
    assert vsn_bypass["repair_task"] == "TFT-SR02"
    assert vsn_bypass["v2_artifact_status"] == "released"
    assert vsn_bypass["integration_kind"] == "additive"

    compression = TFT_EXTENSION_MIGRATION_CAPABILITIES["temporal_compression"]
    assert compression["repair_task"] == "TFT-SR07"
    assert compression["v2_artifact_status"] == "pending_repair"
    assert compression["integration_kind"] == "pending_structural_repair"

    moe = _args(tft_use_regime_moe=True)
    compression_args = _args(tft_use_temporal_compression=True)
    assert pending_v2_artifact_extensions(moe) == []
    assert validate_tft_v2_artifact_readiness(moe) is moe
    assert pending_v2_artifact_extensions(compression_args) == [
        "temporal_compression"
    ]
    with pytest.raises(RuntimeError, match="TFT-SR07"):
        validate_tft_v2_artifact_readiness(compression_args)


def test_resolution_is_idempotent_and_paired_reference_can_disable_branch():
    variant = _args(tft_use_fft_branch=True)
    first = resolve_tft_extension_modes(variant)
    second = resolve_tft_extension_modes(variant)
    assert first == second

    reference = deepcopy(variant)
    reference.tft_use_fft_branch = False
    reference = apply_tft_profile(reference)
    assert reference.tft_fft_integration_mode == "off"
    assert variant.tft_fft_integration_mode == "neutral"


def test_experimental_profile_resolves_every_registered_branch_neutral_in_v2():
    args = _args(tft_profile="experimental_full")
    assert set(args.tft_resolved_extension_modes) == set(
        TFT_EXTENSION_INTEGRATION_SPECS
    )
    assert set(args.tft_resolved_extension_modes.values()) == {"neutral"}


def test_unknown_extension_lookup_fails_explicitly():
    with pytest.raises(KeyError, match="Unknown TFT extension"):
        get_tft_extension_mode(_args(), "not_an_extension")
