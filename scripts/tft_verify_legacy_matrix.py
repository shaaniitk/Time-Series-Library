#!/usr/bin/env python3
"""Emit or verify the immutable July-2026 native-TFT legacy-v1 matrix manifest."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "metadata/tft/legacy_v1_matrix_manifest.json"
MATRIX_SCRIPT = Path("scripts/long_term_forecast/ETT_script/TFT_ETTh1_OT_feature_matrix.sh")
PRODUCER_SCRIPT = Path("metadata/tft/TFT_ETTh1_OT_feature_matrix_legacy_producer.sh")
DATASET = Path("dataset/ETT-small/ETTh1.csv")

CASE_OVERRIDES = {
    "tft_ot_p24_baseline": {},
    "tft_ot_p24_joint_quantile": {
        "tft_use_quantile_head": True,
        "tft_output_mode": "joint",
    },
    "tft_ot_p24_quantile_only": {
        "tft_use_quantile_head": True,
        "tft_output_mode": "quantile",
        "loss": "Quantile",
    },
    "tft_ot_p24_alibi": {"tft_attention_position_bias": "alibi"},
    "tft_ot_p24_sdpa": {"tft_attention_backend": "sdpa"},
    "tft_ot_p24_xattn_interp": {
        "tft_use_explicit_cross_attention": True,
        "tft_cross_attention_type": "interpretable",
    },
    "tft_ot_p24_lag": {
        "tft_use_lag_attention": True,
        "tft_lag_scales": [1, 2, 4],
    },
    "tft_ot_p24_crossmix_sparse": {
        "tft_cross_variable_mixing": True,
        "tft_graph_type": "sparse",
        "tft_graph_top_k": 3,
        "tft_graph_num_layers": 2,
    },
    "tft_ot_p24_fft": {
        "tft_use_fft_branch": True,
        "tft_fft_modes": 16,
        "tft_fft_mode_select": "learned",
    },
    "tft_ot_p24_higher_order": {
        "tft_use_higher_order": True,
        "tft_interaction_order": 2,
        "tft_interaction_rank": 8,
    },
    "tft_ot_p24_moe": {
        "tft_use_regime_moe": True,
        "tft_num_regimes": 3,
        "tft_num_moe_experts": 4,
        "tft_moe_top_k": 2,
        "tft_moe_hidden_size": 16,
        "tft_moe_aux_loss_coeff": 0.02,
    },
    "tft_ot_p24_temporal_compression": {
        "tft_use_temporal_compression": True,
        "tft_tc_stride": 2,
        "tft_tc_threshold": 64,
    },
    "tft_ot_p24_covariate_reattention": {
        "tft_cross_variable_mixing": True,
        "tft_graph_type": "sparse",
        "tft_graph_top_k": 3,
        "tft_covariate_reattention": True,
    },
    "tft_ot_p24_experimental_profile": {
        "tft_profile": "experimental_full",
        "d_model": 16,
        "n_heads": 2,
        "dropout": 0.25,
        "learning_rate": 3e-5,
        "tft_moe_hidden_size": 16,
        "tft_interaction_rank": 8,
        "tft_fft_modes": 16,
        "tft_graph_top_k": 3,
        "tft_tc_threshold": 64,
    },
}

COMMON_CONFIG = {
    "task_name": "long_term_forecast",
    "data": "ETTh1",
    "features": "MS",
    "target": "OT",
    "freq": "h",
    "seq_len": 96,
    "label_len": 48,
    "pred_len": 24,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 1,
    "tft_target_pos": [6],
    "d_model": 16,
    "n_heads": 2,
    "e_layers": 1,
    "d_layers": 1,
    "dropout": 0.2,
    "learning_rate": 5e-5,
    "lradj": "cosine",
    "train_epochs": 30,
    "batch_size": 64,
    "patience": 6,
    "loss": "MSE",
    "tft_profile": "extended_safe",
    "tft_temporal_backbone": "lstm",
    "tft_temporal_backbone_layers": 1,
    "tft_temporal_kernel_size": 3,
    "tft_attention_dropout": 0.08,
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path):
    path = Path(path)
    absolute = path if path.is_absolute() else REPOSITORY_ROOT / path
    return {
        "path": absolute.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": absolute.stat().st_size,
        "sha256": _sha256(absolute),
    }


def _repository_head():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_manifest():
    result_dirs = sorted(
        path
        for path in (REPOSITORY_ROOT / "results").glob(
            "long_term_forecast_tft_ot_p24_*"
        )
        if "_tsv" not in path.name
    )
    checkpoint_dirs = {
        path.name: path
        for path in (REPOSITORY_ROOT / "checkpoints").glob(
            "long_term_forecast_tft_ot_p24_*"
        )
        if "_tsv" not in path.name
    }
    cases = []
    setting_pattern = re.compile(
        r"long_term_forecast_(.*?)_TemporalFusionTransformer_.*_tp(.*)_td([0-9a-f]+)$"
    )
    for result_dir in result_dirs:
        match = setting_pattern.match(result_dir.name)
        if match is None:
            raise RuntimeError(f"Unrecognized TFT matrix setting: {result_dir.name}")
        case_id, profile, config_digest = match.groups()
        if case_id not in CASE_OVERRIDES:
            raise RuntimeError(f"No frozen override record for {case_id}")
        checkpoint_dir = checkpoint_dirs.get(result_dir.name)
        if checkpoint_dir is None:
            raise RuntimeError(f"Missing checkpoint directory for {result_dir.name}")

        values = np.load(result_dir / "metrics.npy").tolist()
        artifacts = [
            _artifact_record(path)
            for path in sorted(result_dir.iterdir())
            if path.is_file()
        ]
        artifacts.append(
            _artifact_record(
                checkpoint_dir / "checkpoint.pth"
            )
        )
        quantile_path = result_dir / "quantile_metrics.json"
        cases.append(
            {
                "case_id": case_id,
                "setting": result_dir.name,
                "profile": profile,
                "config_digest": config_digest,
                "overrides": CASE_OVERRIDES[case_id],
                "metrics": dict(
                    zip(("mae", "mse", "rmse", "mape", "mspe"), values)
                ),
                "quantile_metrics": (
                    json.loads(quantile_path.read_text(encoding="utf-8"))
                    if quantile_path.exists()
                    else None
                ),
                "artifacts": artifacts,
            }
        )

    if len(cases) != 14 or len({case["case_id"] for case in cases}) != 14:
        raise RuntimeError("Legacy matrix inventory must contain 14 unique cases")

    return {
        "manifest_schema_version": 1,
        "matrix_id": "etth1_tft_p24_legacy_v1_2026_07",
        "extension_semantics_version": 1,
        "case_count": 14,
        "artifact_mutation_policy": (
            "immutable; this tracked manifest labels and hashes ignored local "
            "artifacts without modifying them"
        ),
        "repository_head_at_inventory": _repository_head(),
        "provenance_caveat": (
            "the matrix ran from a dirty worktree before this manifest existed; "
            "the exact pre-run worktree diff was not frozen"
        ),
        "producer_script": _artifact_record(PRODUCER_SCRIPT),
        "explicit_v1_replay_script": _artifact_record(MATRIX_SCRIPT),
        "dataset": _artifact_record(DATASET),
        "effective_seed": 2021,
        "seed_caveat": (
            "run.py ignored parsed --seed and hardcoded 2021; optional modules "
            "also changed downstream RNG consumption, so cases are not paired ablations"
        ),
        "common_config": COMMON_CONFIG,
        "cases": cases,
    }


def verify_manifest(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("extension_semantics_version") != 1:
        raise RuntimeError("Legacy matrix manifest must declare semantics version 1")
    cases = manifest.get("cases", [])
    if manifest.get("case_count") != 14 or len(cases) != 14:
        raise RuntimeError("Legacy matrix manifest does not contain 14 cases")
    if len({case["case_id"] for case in cases}) != 14:
        raise RuntimeError("Legacy matrix manifest case IDs are not unique")

    records = [
        manifest["producer_script"],
        manifest["explicit_v1_replay_script"],
        manifest["dataset"],
    ]
    records.extend(
        artifact for case in cases for artifact in case.get("artifacts", [])
    )
    failures = []
    for record in records:
        path = REPOSITORY_ROOT / record["path"]
        if not path.exists():
            failures.append(f"missing: {record['path']}")
            continue
        if path.stat().st_size != record["bytes"]:
            failures.append(f"size: {record['path']}")
            continue
        if _sha256(path) != record["sha256"]:
            failures.append(f"sha256: {record['path']}")
    if failures:
        raise RuntimeError("Legacy matrix verification failed:\n" + "\n".join(failures))
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emit",
        action="store_true",
        help="Print a freshly inventoried manifest to stdout without writing it.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Tracked manifest to verify.",
    )
    args = parser.parse_args()
    if args.emit:
        print(json.dumps(build_manifest(), indent=2, sort_keys=True))
        return
    count = verify_manifest(args.manifest.resolve())
    print(f"verified {count} immutable legacy-v1 artifacts")


if __name__ == "__main__":
    main()
