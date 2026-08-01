import copy
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import (
    TFT_REPRODUCIBILITY_FILENAME,
    Exp_Long_Term_Forecast,
)
from models.TemporalFusionTransformer import Model as NativeTFT
from run import (
    build_parser,
    build_setting,
    execute_training_runs,
    normalize_args,
    resolve_run_args,
)
from utils.reproducibility import (
    build_paired_models,
    isolated_rng,
    stable_json_hash,
    state_dict_sha256,
)


def _cli_args(*extra):
    argv = (
        "--task_name long_term_forecast --is_training 1 "
        "--model_id reproducibility-test --model TemporalFusionTransformer "
        "--data ETTh1 --features MS --seq_len 24 --label_len 12 --pred_len 4 "
        "--enc_in 7 --dec_in 7 --c_out 1 --tft_target_pos 6 "
        "--d_model 8 --n_heads 2 --e_layers 1 --d_layers 1 "
        "--tft_profile extended_safe --tft_temporal_backbone lstm "
        "--tft_declared_regular_sampling "
        "--train_epochs 1 --batch_size 2 --num_workers 0 --no_use_gpu"
    ).split()
    return normalize_args(build_parser().parse_args(argv + list(extra)))


def test_v2_defaults_to_validation_only_and_legacy_preserves_test_policy():
    current = _cli_args()
    legacy = _cli_args("--tft_extension_semantics_version", "1")
    assert current.evaluation_policy == "validation_only"
    assert current.deterministic_mode == "warn"
    assert legacy.evaluation_policy == "legacy_val_and_test"
    assert legacy.deterministic_mode == "off"


def test_iteration_seed_schedule_and_run_settings_are_stable_and_distinct():
    args = _cli_args("--seed", "17")
    first, first_bundle = resolve_run_args(args, 0)
    repeat, repeat_bundle = resolve_run_args(args, 0)
    second, second_bundle = resolve_run_args(args, 1)

    assert first_bundle == repeat_bundle
    assert first.reproducibility_digest == repeat.reproducibility_digest
    assert first_bundle != second_bundle
    assert first.reproducibility_digest != second.reproducibility_digest
    assert build_setting(first, 0) == build_setting(repeat, 0)
    assert build_setting(first, 0) != build_setting(second, 1)
    assert "_rd" in build_setting(first, 0)

    legacy = _cli_args("--tft_extension_semantics_version", "1")
    legacy.seed = 2021
    legacy_run, legacy_bundle = resolve_run_args(legacy, 0)
    assert "_rd" not in build_setting(legacy_run, 0)
    assert legacy_bundle.derivation_version == 0
    assert {
        legacy_bundle.model_init_seed,
        legacy_bundle.extension_init_seed,
        legacy_bundle.data_order_seed,
        legacy_bundle.worker_seed,
        legacy_bundle.training_seed,
    } == {2021}
    assert legacy_run._isolated_rng_streams is False


def test_runner_suppresses_automatic_test_under_validation_only():
    calls = []

    class DummyExp:
        def __init__(self, args):
            self.args = args
            calls.append(("init", args.run_index))

        def train(self, setting):
            calls.append(("train", setting))

        def test(self, setting, test=0):
            calls.append(("test", setting, test))

    settings = execute_training_runs(_cli_args("--itr", "2"), DummyExp)
    assert len(settings) == 2
    assert len(set(settings)) == 2
    assert [call[0] for call in calls].count("train") == 2
    assert not any(call[0] == "test" for call in calls)

    calls.clear()
    execute_training_runs(
        _cli_args(
            "--tft_extension_semantics_version",
            "1",
            "--evaluation_policy",
            "legacy_val_and_test",
        ),
        DummyExp,
    )
    assert [call[0] for call in calls].count("test") == 1


def test_actual_tft_fft_pair_has_bitwise_identical_shared_state():
    reference_args = _cli_args("--tft_extension_semantics_version", "1")
    variant_args = _cli_args(
        "--tft_extension_semantics_version",
        "1",
        "--tft_use_fft_branch",
        "--tft_fft_modes",
        "4",
    )
    reference, variant, report = build_paired_models(
        lambda: NativeTFT(copy.deepcopy(reference_args)),
        lambda: NativeTFT(copy.deepcopy(variant_args)),
        reference_seed=101,
        variant_seed=202,
        allowed_variant_only_regexes=(
            r"\.fft_branch\.",
            r"\.fft_fusion_gate\.",
        ),
    )

    reference_state = reference.state_dict()
    variant_state = variant.state_dict()
    assert report["copied_tensor_count"] > 0
    assert report["shared_state_hash"] == report["shared_state_sha256"]
    assert report["variant_only_names"]
    for name in report["copied_names"]:
        assert torch.equal(reference_state[name], variant_state[name]), name


def test_production_paired_fft_construction_is_bound_to_model_state():
    args = _cli_args(
        "--tft_use_fft_branch",
        "--tft_fft_modes",
        "4",
        "--tft_paired_initialization",
        "--tft_paired_reference_disable",
        "tft_use_fft_branch",
    )
    run_args, bundle = resolve_run_args(args, 0)
    experiment = Exp_Long_Term_Forecast(run_args)
    report = run_args._paired_initialization_report

    assert report["binding_verified"] is True
    assert report["variant_state_sha256"] == experiment._initial_state_hash
    assert report["bound_initial_state_sha256"] == experiment._initial_state_hash
    assert report["shared_state_sha256"]
    assert report["copied_tensor_count"] > 0
    assert report["variant_only_names"]
    assert all(
        ".fft_branch." in name
        or ".fft_fusion_gate." in name
        or ".fft_residual_adapter." in name
        for name in report["variant_only_names"]
    )
    assert report["reference_config_digest"] != report["variant_config_digest"]
    assert (
        report["paired_reference_spec_digest"]
        == run_args.tft_paired_reference_spec_digest
    )

    manifest = experiment._build_reproducibility_manifest("paired-test")
    assert manifest["shared_state_hash"] == report["shared_state_sha256"]
    assert manifest["paired_initialization"]["binding_verified"] is True
    assert manifest["paired_initialization_report_digest"] == stable_json_hash(
        manifest["paired_initialization"]
    )
    assert (
        manifest["paired_reference_spec_digest"]
        == run_args.tft_paired_reference_spec_digest
    )

    reference_args = _cli_args("--tft_fft_modes", "4")
    with isolated_rng(bundle.model_init_seed):
        standalone_reference = NativeTFT(reference_args).float()
    assert (
        state_dict_sha256(standalone_reference.state_dict())
        == report["reference_state_sha256"]
    )


def test_paired_reference_contract_changes_run_identity_and_rejects_spoofing():
    variant = _cli_args("--tft_use_fft_branch", "--tft_fft_modes", "4")
    paired = _cli_args(
        "--tft_use_fft_branch",
        "--tft_fft_modes",
        "4",
        "--tft_paired_initialization",
        "--tft_paired_reference_disable",
        "tft_use_fft_branch",
    )
    variant_run, _ = resolve_run_args(variant, 0)
    paired_run, _ = resolve_run_args(paired, 0)

    assert variant_run.tft_config_digest == paired_run.tft_config_digest
    assert variant_run.reproducibility_digest != paired_run.reproducibility_digest
    assert build_setting(variant_run, 0) != build_setting(paired_run, 0)

    experiment = Exp_Long_Term_Forecast(paired_run)
    paired_run._paired_initialization_report["variant_state_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="mutated after model construction"):
        experiment._build_reproducibility_manifest("tampered")

    unpaired_with_spoof = copy.deepcopy(variant_run)
    unpaired_with_spoof._paired_initialization_report = {"shared_state_hash": "fake"}
    clean_experiment = Exp_Long_Term_Forecast(unpaired_with_spoof)
    assert not hasattr(clean_experiment.args, "_paired_initialization_report")


@pytest.mark.parametrize(
    "mutation",
    (
        "shared_hash",
        "reference_hash",
        "copied_names",
        "unmatched_names",
        "report_pair_spec",
        "args_pair_controls",
        "args_pair_spec",
        "args_pair_digest",
    ),
)
def test_every_paired_report_and_reference_binding_is_tamper_evident(mutation):
    args = _cli_args(
        "--tft_use_fft_branch",
        "--tft_fft_modes",
        "4",
        "--tft_paired_initialization",
        "--tft_paired_reference_disable",
        "tft_use_fft_branch",
    )
    run_args, _ = resolve_run_args(args, 0)
    experiment = Exp_Long_Term_Forecast(run_args)
    report = run_args._paired_initialization_report
    if mutation == "shared_hash":
        report["shared_state_sha256"] = "forged"
    elif mutation == "reference_hash":
        report["reference_state_sha256"] = "forged"
    elif mutation == "copied_names":
        report["copied_names"].append("forged.tensor")
    elif mutation == "unmatched_names":
        report["variant_only_names"].append("forged.tensor")
    elif mutation == "report_pair_spec":
        report["paired_reference_spec"]["reference_disable"] = []
    elif mutation == "args_pair_controls":
        run_args.tft_paired_reference_overrides = {"tft_debug_checks": True}
    elif mutation == "args_pair_spec":
        run_args.tft_paired_reference_spec["reference_disable"] = []
    elif mutation == "args_pair_digest":
        run_args.tft_paired_reference_spec_digest = "forged"

    with pytest.raises(RuntimeError, match="mutated|inconsistent"):
        experiment._build_reproducibility_manifest("tampered")


@pytest.mark.parametrize(
    "extra, message",
    (
        (
            (
                "--tft_paired_reference_disable",
                "tft_use_fft_branch",
            ),
            "require --tft_paired_initialization",
        ),
        (
            (
                "--tft_extension_semantics_version",
                "1",
                "--tft_use_fft_branch",
                "--tft_paired_initialization",
                "--tft_paired_reference_disable",
                "tft_use_fft_branch",
            ),
            "requires --tft_extension_semantics_version 2",
        ),
        (
            (
                "--tft_paired_initialization",
                "--tft_paired_reference_disable",
                "tft_use_fft_branch",
            ),
            "does not enable",
        ),
    ),
)
def test_invalid_paired_cli_contracts_fail_closed(extra, message):
    with pytest.raises(ValueError, match=message):
        _cli_args(*extra)


def test_paired_reference_cannot_change_schema_or_non_architecture_controls():
    with pytest.raises(ValueError, match="not an allowed TFT architecture control"):
        _cli_args(
            "--tft_paired_initialization",
            "--tft_use_fft_branch",
            "--tft_paired_reference_disable",
            "tft_use_fft_branch",
            "--tft_paired_reference_overrides",
            '{"tft_target_pos": [5]}',
        )
    with pytest.raises(ValueError, match="not an allowed TFT architecture control"):
        _cli_args(
            "--tft_paired_initialization",
            "--tft_use_fft_branch",
            "--tft_debug_checks",
            "--tft_paired_reference_disable",
            "tft_debug_checks",
        )

    # Exp validates the resolved schema independently of CLI normalization, so
    # mutating a previously valid Namespace cannot bypass the contract.
    args = _cli_args(
        "--tft_paired_initialization",
        "--tft_use_fft_branch",
        "--tft_paired_reference_disable",
        "tft_use_fft_branch",
    )
    run_args, _ = resolve_run_args(args, 0)
    run_args.tft_paired_reference_overrides = {"tft_target_pos": [5]}
    with pytest.raises(ValueError, match="controls no longer match"):
        Exp_Long_Term_Forecast(run_args)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("tft_extension_semantics_version", 1),
        ("tft_debug_checks", True),
        ("tft_point_loss_coeff", 2.0),
        ("tft_fft_integration_mode", "small_residual"),
        ("tft_extension_residual_shape", "channel"),
        ("tft_small_residual_init", 0.01),
    ),
)
def test_exp_rejects_resealed_forbidden_paired_reference_overrides(key, value):
    args = _cli_args(
        "--tft_paired_initialization",
        "--tft_use_fft_branch",
        "--tft_paired_reference_disable",
        "tft_use_fft_branch",
    )
    run_args, _ = resolve_run_args(args, 0)
    run_args.tft_paired_reference_overrides = {key: value}
    run_args.tft_paired_reference_spec["reference_overrides"] = {key: value}
    run_args.tft_paired_reference_spec_digest = stable_json_hash(
        run_args.tft_paired_reference_spec
    )[:12]
    with pytest.raises(ValueError, match="not an allowed architecture control"):
        Exp_Long_Term_Forecast(run_args)


def test_exp_rejects_stale_paired_architecture_spec_after_namespace_mutation():
    args = _cli_args(
        "--tft_paired_initialization",
        "--tft_use_fft_branch",
        "--tft_paired_reference_disable",
        "tft_use_fft_branch",
    )
    run_args, _ = resolve_run_args(args, 0)
    run_args.tft_paired_reference_disable = ["tft_use_higher_order"]
    with pytest.raises(ValueError, match="controls no longer match"):
        Exp_Long_Term_Forecast(run_args)


def test_production_v2_micro_run_replays_checkpoint_and_order_hashes(tmp_path):
    data_path = tmp_path / "series.csv"
    rows = 100
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="h"),
            "feature": np.sin(np.arange(rows) / 7.0),
            "OT": np.cos(np.arange(rows) / 11.0),
        }
    ).to_csv(data_path, index=False)

    argv = (
        "--task_name long_term_forecast --is_training 1 "
        "--model_id sr01-micro --model TemporalFusionTransformer "
        f"--data custom --root_path {tmp_path} --data_path series.csv "
        "--features MS --target OT --freq h "
        "--seq_len 8 --label_len 4 --pred_len 2 "
        "--enc_in 2 --dec_in 2 --c_out 1 "
        "--tft_observed_pos 0,1 --tft_target_pos 1 "
        "--d_model 8 --n_heads 2 --e_layers 1 --d_layers 1 "
        "--tft_profile extended_safe --tft_temporal_backbone lstm "
        "--tft_temporal_backbone_layers 1 --dropout 0.0 "
        "--tft_declared_regular_sampling "
        "--tft_vsn_residual_bypass --tft_paired_initialization "
        "--tft_paired_reference_disable tft_vsn_residual_bypass "
        "--train_epochs 1 --batch_size 8 --num_workers 0 --patience 1 "
        f"--checkpoints {tmp_path / 'checkpoints'} --seed 314 --no_use_gpu"
    ).split()
    args = normalize_args(build_parser().parse_args(argv))

    # A separately configured baseline and the paired variant must resolve the
    # same fold and exact sample-ID order even though their run identities and
    # architecture digests differ.
    baseline_args = copy.deepcopy(args)
    baseline_args.tft_vsn_residual_bypass = False
    baseline_args.tft_paired_initialization = False
    baseline_args.tft_paired_reference_disable = []
    baseline_args.tft_paired_reference_overrides = {}
    baseline_args.tft_paired_reference_only_pattern = []
    baseline_args.tft_paired_variant_only_pattern = []
    baseline_args = normalize_args(baseline_args)
    baseline_run, _ = resolve_run_args(baseline_args, 0)
    variant_run, _ = resolve_run_args(args, 0)
    _, baseline_loader = data_provider(baseline_run, "train")
    _, variant_loader = data_provider(variant_run, "train")
    assert list(baseline_loader.sampler) == list(variant_loader.sampler)
    assert (
        baseline_loader.reproducibility_metadata["fold_manifest_hash"]
        == variant_loader.reproducibility_metadata["fold_manifest_hash"]
    )
    assert (
        baseline_loader.reproducibility_metadata["epoch0_order_hash"]
        == variant_loader.reproducibility_metadata["epoch0_order_hash"]
    )
    assert (
        baseline_loader.reproducibility_metadata["first_batch_index_hash"]
        == variant_loader.reproducibility_metadata["first_batch_index_hash"]
    )

    setting = execute_training_runs(args, Exp_Long_Term_Forecast)[0]
    checkpoint_dir = Path(args.checkpoints) / setting
    checkpoint = checkpoint_dir / "checkpoint.pth"
    manifest_path = checkpoint_dir / TFT_REPRODUCIBILITY_FILENAME
    first_checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    repeated_setting = execute_training_runs(args, Exp_Long_Term_Forecast)[0]
    second_checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    second_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert repeated_setting == setting
    assert first_checkpoint_hash == second_checkpoint_hash
    assert first_manifest["initial_state_hash"] == second_manifest["initial_state_hash"]
    assert first_manifest["final_state_hash"] == second_manifest["final_state_hash"]
    assert first_manifest["first_batch_index_hash"] == second_manifest["first_batch_index_hash"]
    assert first_manifest["fold_manifest_hash"] == second_manifest["fold_manifest_hash"]
    assert first_manifest["shared_state_hash"]
    assert first_manifest["shared_state_hash"] == second_manifest["shared_state_hash"]
    assert first_manifest["paired_initialization"]["binding_verified"] is True
    assert first_manifest["evaluation_policy"] == "validation_only"
    assert first_manifest["loaders"]["test"] is None

    changed_args = copy.deepcopy(args)
    changed_args.seed = 315
    changed_setting = execute_training_runs(
        changed_args, Exp_Long_Term_Forecast
    )[0]
    changed_manifest_path = (
        Path(changed_args.checkpoints)
        / changed_setting
        / TFT_REPRODUCIBILITY_FILENAME
    )
    changed_manifest = json.loads(changed_manifest_path.read_text(encoding="utf-8"))
    assert changed_setting != setting
    assert changed_manifest["initial_state_hash"] != first_manifest["initial_state_hash"]
    assert (
        changed_manifest["loaders"]["train"]["epoch0_order_hash"]
        != first_manifest["loaders"]["train"]["epoch0_order_hash"]
    )
    assert (
        changed_manifest["first_batch_index_hash"]
        != first_manifest["first_batch_index_hash"]
    )


def test_long_term_experiment_does_not_globally_suppress_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("sr01-warning-sentinel", UserWarning)
    assert any("sr01-warning-sentinel" in str(item.message) for item in caught)
