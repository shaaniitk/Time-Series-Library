import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from run import build_parser, build_setting, normalize_args
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from models.TemporalFusionTransformer import Model as NativeTFT
from utils.tft_config import (
    TFT_CHECKPOINT_METADATA_FILENAME,
    TFT_CURRENT_EXTENSION_SEMANTICS_VERSION,
    TFT_LEGACY_EXTENSION_SEMANTICS_VERSION,
    apply_tft_profile,
    build_tft_semantics_manifest,
    read_tft_semantics_metadata,
    validate_tft_checkpoint_compatibility,
    validate_tft_v2_artifact_readiness,
    write_tft_semantics_metadata,
)
from utils.tft_checkpoint import (
    legacy_setting_id,
    load_tft_checkpoint,
    resolve_tft_checkpoint_path,
)
from scripts.tft_verify_legacy_matrix import verify_manifest
from utils.tools import EarlyStopping


def _args(**overrides):
    values = {
        "model": "TemporalFusionTransformer",
        "task_name": "long_term_forecast",
        "model_id": "semantics-test",
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
        "expand": 2,
        "d_conv": 4,
        "factor": 1,
        "embed": "timeF",
        "distil": True,
        "des": "test",
        "dropout": 0.1,
        "freq": "h",
        "tft_profile": "extended_safe",
    }
    values.update(overrides)
    return apply_tft_profile(SimpleNamespace(**values))


def test_semantics_v2_is_default_and_material_to_digest_and_setting():
    current = _args()
    legacy = _args(tft_extension_semantics_version=1)

    assert current.tft_extension_semantics_version == TFT_CURRENT_EXTENSION_SEMANTICS_VERSION == 2
    assert legacy.tft_extension_semantics_version == TFT_LEGACY_EXTENSION_SEMANTICS_VERSION == 1
    assert current.tft_config_digest != legacy.tft_config_digest
    assert "_tsv2_" in build_setting(current, 0)
    assert "_tsv1_" in build_setting(legacy, 0)


def test_semantics_version_rejects_unknown_values():
    with pytest.raises(ValueError, match="tft_extension_semantics_version"):
        _args(tft_extension_semantics_version=3)


def test_legacy_matrix_baseline_digest_is_frozen():
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
    assert args.tft_digest_schema == "legacy-v1"
    assert args.tft_config_digest == "6c54053ec5d7"


def test_manifest_roundtrip_records_capabilities_and_active_extensions(tmp_path):
    args = _args(
        tft_extension_semantics_version=1,
        tft_use_lag_attention=True,
    )
    torch.save({}, tmp_path / "checkpoint.pth")
    path = write_tft_semantics_metadata(tmp_path, args, artifact_kind="checkpoint")

    assert path == tmp_path / TFT_CHECKPOINT_METADATA_FILENAME
    loaded = read_tft_semantics_metadata(tmp_path / "checkpoint.pth")
    assert loaded["extension_semantics_version"] == 1
    assert loaded["config_digest"] == args.tft_config_digest
    assert loaded["active_extensions"]["lag_attention"] is True
    assert loaded["migration_capabilities"]["lag_attention"]["v1_to_v2"] == "retrain"
    assert loaded["checkpoint"]["sha256"] == hashlib.sha256(
        (tmp_path / "checkpoint.pth").read_bytes()
    ).hexdigest()
    assert json.loads(path.read_text(encoding="utf-8")) == loaded


def test_checkpoint_without_metadata_is_classified_as_legacy(tmp_path):
    assert read_tft_semantics_metadata(tmp_path / "checkpoint.pth") is None

    args = _args()
    with pytest.raises(RuntimeError, match="cannot be mapped into semantics v2"):
        validate_tft_checkpoint_compatibility(args, tmp_path / "checkpoint.pth")


def test_legacy_extension_weights_never_silently_enter_v2(tmp_path):
    args = _args(
        tft_use_lag_attention=True,
        tft_allow_legacy_extension_checkpoint=True,
    )
    with pytest.raises(RuntimeError, match="cannot be mapped into semantics v2"):
        validate_tft_checkpoint_compatibility(args, tmp_path / "checkpoint.pth")


def test_legacy_extension_replay_requires_version_and_explicit_flag(tmp_path):
    without_flag = _args(
        tft_extension_semantics_version=1,
        tft_use_lag_attention=True,
    )
    with pytest.raises(RuntimeError, match="tft_allow_legacy_extension_checkpoint"):
        validate_tft_checkpoint_compatibility(without_flag, tmp_path / "checkpoint.pth")

    with_flag = _args(
        tft_extension_semantics_version=1,
        tft_allow_legacy_extension_checkpoint=True,
        tft_use_lag_attention=True,
    )
    resolved = validate_tft_checkpoint_compatibility(with_flag, tmp_path / "checkpoint.pth")
    assert resolved["load_mode"] == "explicit_legacy_replay"


def test_trusted_current_run_can_reload_its_own_v1_checkpoint(tmp_path):
    args = _args(tft_extension_semantics_version=1, tft_use_lag_attention=True)
    torch.save({}, tmp_path / "checkpoint.pth")
    write_tft_semantics_metadata(tmp_path, args, artifact_kind="checkpoint")
    resolved = validate_tft_checkpoint_compatibility(
        args,
        tmp_path / "checkpoint.pth",
        external_load=False,
    )
    assert resolved["load_mode"] == "trusted_current_run_legacy_reload"


def test_v2_checkpoint_metadata_rejects_legacy_runtime(tmp_path):
    current = _args()
    torch.save({}, tmp_path / "checkpoint.pth")
    write_tft_semantics_metadata(tmp_path, current, artifact_kind="checkpoint")

    legacy_runtime = _args(tft_extension_semantics_version=1)
    with pytest.raises(RuntimeError, match="checkpoint semantics version 2"):
        validate_tft_checkpoint_compatibility(
            legacy_runtime,
            tmp_path / "checkpoint.pth",
        )


def test_manifest_builder_is_deterministic():
    args = _args(tft_use_fft_branch=True)
    left = build_tft_semantics_manifest(args, artifact_kind="result")
    right = build_tft_semantics_manifest(args, artifact_kind="result")
    assert left == right


def test_profile_application_is_idempotent():
    args = _args(tft_use_fft_branch=True)
    first_digest = args.tft_config_digest
    first_manifest = build_tft_semantics_manifest(args)
    reapplied = apply_tft_profile(args)
    assert reapplied.tft_config_digest == first_digest
    assert build_tft_semantics_manifest(reapplied) == first_manifest


def test_v2_bare_state_dict_and_sidecar_roundtrip(tmp_path):
    args = _args()
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(source.state_dict(), checkpoint)
    write_tft_semantics_metadata(tmp_path, args, artifact_kind="checkpoint")

    policy = load_tft_checkpoint(
        target,
        checkpoint,
        args,
        map_location="cpu",
    )
    assert policy["load_mode"] == "current"
    assert all(
        torch.equal(left, right)
        for left, right in zip(source.parameters(), target.parameters())
    )
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(loaded) == set(source.state_dict())


@pytest.mark.parametrize("checkpoint_wrapped", [False, True])
def test_v2_checkpoint_is_portable_across_data_parallel_wrapper(
    tmp_path,
    checkpoint_wrapped,
):
    args = _args()
    source_plain = torch.nn.Linear(3, 2)
    target_plain = torch.nn.Linear(3, 2)
    source = (
        torch.nn.DataParallel(source_plain) if checkpoint_wrapped else source_plain
    )
    target = (
        target_plain if checkpoint_wrapped else torch.nn.DataParallel(target_plain)
    )
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(source.state_dict(), checkpoint)
    write_tft_semantics_metadata(tmp_path, args, artifact_kind="checkpoint")

    load_tft_checkpoint(target, checkpoint, args, map_location="cpu")

    assert all(
        torch.equal(left.detach().cpu(), right.detach().cpu())
        for left, right in zip(source_plain.parameters(), target_plain.parameters())
    )


def test_unrepaired_extension_cannot_be_stamped_as_v2(tmp_path):
    args = _args(tft_covariate_reattention=True)
    torch.save({}, tmp_path / "checkpoint.pth")
    with pytest.raises(RuntimeError, match="Refusing to stamp"):
        write_tft_semantics_metadata(tmp_path, args, artifact_kind="checkpoint")
    with pytest.raises(RuntimeError, match="covariate_reattention"):
        validate_tft_v2_artifact_readiness(args)


def test_production_training_rejects_pending_v2_extension_before_side_effects(
    tmp_path,
):
    args = _args(
        tft_covariate_reattention=True,
        checkpoints=str(tmp_path / "checkpoints"),
    )
    experiment = object.__new__(Exp_Long_Term_Forecast)
    experiment.args = args
    experiment._get_data = lambda **_: pytest.fail(
        "data loading must not begin before v2 readiness validation"
    )

    with pytest.raises(RuntimeError, match="covariate_reattention"):
        experiment.train("must-not-exist")

    assert not (tmp_path / "checkpoints").exists()


def test_each_atomic_best_checkpoint_can_be_stamped_immediately(tmp_path):
    args = _args()
    model = torch.nn.Linear(3, 2)
    callback_paths = []

    def stamp(checkpoint_path):
        callback_paths.append(Path(checkpoint_path))
        write_tft_semantics_metadata(
            tmp_path,
            args,
            artifact_kind="checkpoint",
            setting="atomic-save-test",
            checkpoint_path=checkpoint_path,
        )

    stopper = EarlyStopping(
        patience=2,
        checkpoint_saved_callback=stamp,
    )
    stopper(1.0, model, str(tmp_path))

    checkpoint = tmp_path / "checkpoint.pth"
    assert callback_paths == [checkpoint]
    metadata = read_tft_semantics_metadata(checkpoint)
    assert metadata["setting"] == "atomic-save-test"
    assert metadata["checkpoint"]["sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()


def test_v2_checkpoint_rejects_config_or_schema_mismatch(tmp_path):
    source_args = _args()
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(torch.nn.Linear(3, 2).state_dict(), checkpoint)
    write_tft_semantics_metadata(tmp_path, source_args, artifact_kind="checkpoint")

    changed_args = _args(dropout=0.25)
    with pytest.raises(RuntimeError, match="config_digest"):
        validate_tft_checkpoint_compatibility(changed_args, checkpoint)


def test_v2_checkpoint_ignores_unused_small_residual_initializer(tmp_path):
    source_args = _args(
        tft_use_regime_moe=True,
        tft_small_residual_init=0.001,
    )
    compatible_args = _args(
        tft_use_regime_moe=True,
        tft_small_residual_init=0.25,
    )
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(torch.nn.Linear(3, 2).state_dict(), checkpoint)
    write_tft_semantics_metadata(tmp_path, source_args, artifact_kind="checkpoint")

    assert source_args.tft_config_digest == compatible_args.tft_config_digest
    policy = validate_tft_checkpoint_compatibility(compatible_args, checkpoint)
    assert policy["load_mode"] == "current"


def test_v2_checkpoint_rejects_hash_mismatch(tmp_path):
    args = _args()
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(torch.nn.Linear(3, 2).state_dict(), checkpoint)
    write_tft_semantics_metadata(tmp_path, args, artifact_kind="checkpoint")
    torch.save(torch.nn.Linear(4, 2).state_dict(), checkpoint)

    with pytest.raises(RuntimeError, match="hash/size"):
        load_tft_checkpoint(torch.nn.Linear(4, 2), checkpoint, args)


def test_missing_checkpoint_reports_file_not_found_before_migration(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_tft_checkpoint(
            torch.nn.Linear(3, 2),
            tmp_path / "missing.pth",
            _args(),
        )


def test_historical_setting_path_is_resolved_only_for_explicit_v1(tmp_path):
    legacy_args = _args(
        tft_extension_semantics_version=1,
        tft_allow_legacy_extension_checkpoint=True,
    )
    versioned_setting = build_setting(legacy_args, 0)
    historical_setting = legacy_setting_id(versioned_setting)
    historical_checkpoint = tmp_path / historical_setting / "checkpoint.pth"
    historical_checkpoint.parent.mkdir(parents=True)
    torch.save({}, historical_checkpoint)

    assert resolve_tft_checkpoint_path(
        tmp_path,
        versioned_setting,
        legacy_args,
    ) == historical_checkpoint
    legacy_args.tft_allow_legacy_extension_checkpoint = False
    assert resolve_tft_checkpoint_path(
        tmp_path,
        versioned_setting,
        legacy_args,
    ) == tmp_path / versioned_setting / "checkpoint.pth"


def test_explicit_v1_native_tft_state_dict_replay_is_exact(tmp_path):
    source_args = _args(
        tft_extension_semantics_version=1,
        tft_allow_legacy_extension_checkpoint=True,
        tft_temporal_backbone="lstm",
    )
    target_args = _args(
        tft_extension_semantics_version=1,
        tft_allow_legacy_extension_checkpoint=True,
        tft_temporal_backbone="lstm",
    )
    source = NativeTFT(source_args)
    target = NativeTFT(target_args)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(source.state_dict(), checkpoint)

    policy = load_tft_checkpoint(target, checkpoint, target_args, map_location="cpu")
    assert policy["load_mode"] == "explicit_legacy_replay"
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, target.state_dict()[name]), name


def test_explicit_v1_bare_state_dict_replay_is_exact(tmp_path):
    args = _args(
        tft_extension_semantics_version=1,
        tft_allow_legacy_extension_checkpoint=True,
        tft_use_lag_attention=True,
    )
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(source.state_dict(), checkpoint)

    policy = load_tft_checkpoint(target, checkpoint, args, map_location="cpu")
    assert policy["load_mode"] == "explicit_legacy_replay"
    assert all(
        torch.equal(left, right)
        for left, right in zip(source.parameters(), target.parameters())
    )


def test_migration_table_covers_every_semantic_repair_component():
    manifest = build_tft_semantics_manifest(_args())
    capabilities = manifest["migration_capabilities"]
    required = {
        "fft_branch",
        "explicit_cross_attention",
        "lag_attention",
        "higher_order_interaction",
        "per_feature_vsn",
        "temporal_compression",
        "graph_cross_mixing",
        "covariate_reattention",
    }
    assert required.issubset(capabilities)
    assert {capabilities[name]["v1_to_v2"] for name in required} == {"retrain"}
    assert all(capabilities[name]["repair_task"].startswith("TFT-SR") for name in required)


def test_tracked_legacy_matrix_manifest_has_14_unique_hashed_cases():
    repository_root = Path(__file__).resolve().parents[1]
    manifest_path = repository_root / "metadata/tft/legacy_v1_matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    assert manifest["extension_semantics_version"] == 1
    assert manifest["case_count"] == len(cases) == 14
    assert len({case["case_id"] for case in cases}) == 14
    assert manifest["producer_script"]["sha256"] == (
        "9a438eb714aaeec8a947a54763092be5a73781c38a6325558cb0cb8f1539b9d2"
    )
    assert manifest["producer_script"] != manifest["explicit_v1_replay_script"]
    for case in cases:
        names = {Path(artifact["path"]).name for artifact in case["artifacts"]}
        assert {"checkpoint.pth", "metrics.npy", "pred.npy", "true.npy"}.issubset(names)
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
            for artifact in case["artifacts"]
        )

    records = [
        manifest["producer_script"],
        manifest["explicit_v1_replay_script"],
        manifest["dataset"],
    ]
    records.extend(
        artifact for case in cases for artifact in case["artifacts"]
    )
    local_artifacts = []
    for record in records:
        path = repository_root / record["path"]
        if not path.exists():
            continue
        local_artifacts.append(path)
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]

    assert len(local_artifacts) >= 3
    if len(local_artifacts) == len(records):
        assert verify_manifest(manifest_path) > 50
