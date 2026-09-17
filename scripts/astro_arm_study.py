#!/usr/bin/env python3
"""Run paired astrology feature arms and write a stop/proceed report.

Synthetic known-answer check (planted Mercury-retrograde effect):
    python scripts/astro_arm_study.py --synthetic --folds F1 --epochs 8

Real study, once the ephemeris package arrives:
    python scripts/astro_arm_study.py \
        --root_path ./data --data_path nifty_market.csv \
        --ephemeris data/ephemeris.parquet --manifest data/ephemeris_manifest.json \
        --folds F1 F2 F3 F4 --seeds 2 3 4
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from astro.report.arm_study import family_importance, train_arm, verdict, write_report  # noqa: E402
from astro.rules.schema import load_ruleset  # noqa: E402
from astro.synthetic_study import write_synthetic_study  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic_effect", type=float, default=0.8)
    parser.add_argument("--work_dir", default="./astro_runs")
    parser.add_argument("--root_path")
    parser.add_argument("--data_path")
    parser.add_argument("--ephemeris")
    parser.add_argument("--manifest")
    parser.add_argument("--ruleset", default="configs/astrology/ast_v1_core.json")
    parser.add_argument("--target", default="log_Close")
    parser.add_argument("--enc_in", type=int, default=4)
    parser.add_argument("--target_pos", type=int, default=3)
    parser.add_argument("--folds", nargs="+", default=["F1", "F2", "F3", "F4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2])
    parser.add_argument("--arms", nargs="*", default=None,
                        help="Defaults to real, zero and every null arm in the ruleset.")
    parser.add_argument("--leave_one_family_out", action="store_true",
                        help="Also retrain one zero:<family> arm per rule family for measured importance.")
    parser.add_argument("--rule_gates", action="store_true",
                        help="Enable learned per-rule input gates (diagnostic importance).")
    parser.add_argument("--prior_coeff", type=float, default=0.0)
    parser.add_argument("--regularity_coeff", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    return parser.parse_args()


def main():
    opts = parse_args()
    os.makedirs(opts.work_dir, exist_ok=True)

    if opts.synthetic:
        study = write_synthetic_study(
            os.path.join(opts.work_dir, "synthetic_data"), effect_size=opts.synthetic_effect
        )
        root_path, data_path = study["root_path"], study["data_path"]
        ephemeris, manifest = study["ephemeris_path"], study["manifest_path"]
    else:
        missing = [n for n in ("root_path", "data_path", "ephemeris", "manifest") if not getattr(opts, n)]
        if missing:
            sys.exit(f"Real studies require: {', '.join('--' + m for m in missing)}")
        root_path, data_path, ephemeris, manifest = (
            opts.root_path, opts.data_path, opts.ephemeris, opts.manifest,
        )

    ruleset = load_ruleset(opts.ruleset)
    arms = opts.arms or ["real", "zero"] + [n.arm_name for n in ruleset.null_arms]
    if opts.leave_one_family_out:
        arms += [f"zero:{family}" for family in ruleset.families]
    observed = ",".join(str(i) for i in range(opts.enc_in))
    base_argv = [
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--model", "TemporalFusionTransformer", "--data", "planetary_market",
        "--root_path", root_path, "--data_path", data_path, "--target", opts.target,
        "--features", "MS", "--enc_in", str(opts.enc_in), "--dec_in", str(opts.enc_in),
        "--c_out", "1", "--tft_observed_pos", observed, "--tft_target_pos", str(opts.target_pos),
        "--freq", "b", "--seq_len", str(opts.seq_len), "--label_len", "16", "--pred_len", "1",
        "--d_model", str(opts.d_model), "--n_heads", "2", "--e_layers", "1", "--d_layers", "1",
        "--dropout", "0.2", "--tft_profile", "extended_safe",
        "--tft_extension_semantics_version", "2", "--tft_declared_regular_sampling",
        "--tft_position_unit", "steps", "--tft_position_source", "row_index",
        "--batch_size", str(opts.batch_size), "--learning_rate", str(opts.learning_rate),
        "--train_epochs", str(opts.epochs), "--patience", str(opts.patience),
        "--astro_ruleset", opts.ruleset,
        "--astro_ephemeris_path", ephemeris, "--astro_ephemeris_manifest", manifest,
        "--astro_prior_coeff", str(opts.prior_coeff),
        "--astro_regularity_coeff", str(opts.regularity_coeff),
    ] + (["--tft_astro_rule_gates"] if opts.rule_gates else [])

    checkpoints = os.path.join(opts.work_dir, "checkpoints")
    results = []
    for fold in opts.folds:
        for seed in opts.seeds:
            for arm in arms:
                print(f"\n=== fold={fold} seed={seed} arm={arm} ===", flush=True)
                result = train_arm(base_argv, arm, fold, seed, checkpoints)
                print(f"    validation MAE {result.mae:.5f}", flush=True)
                results.append(result)
                result.frame.to_csv(
                    os.path.join(opts.work_dir, f"pred_{fold}_s{seed}_{''.join(c if c.isalnum() else '_' for c in arm)}.csv"),
                    index=False,
                )

    primary = [r for r in results if not r.arm.startswith("zero:")]
    report = verdict(primary)
    if opts.leave_one_family_out:
        report["family_importance"] = family_importance(results)
    report["arms"] = arms
    report["synthetic"] = bool(opts.synthetic)
    path = os.path.join(opts.work_dir, "verdict.json")
    write_report(report, path)
    print(f"\nDECISION: {report['decision']}  ({report['rule']})")
    for row in report["rows"]:
        print(
            f"  {row['fold']} s{row['seed']}: real {row['mae_real']:.5f} | zero {row['mae_zero']:.5f}"
            + (f" | best null {row['mae_best_null']:.5f} ({row['best_null']})" if "mae_best_null" in row else "")
        )
    for row in report.get("family_importance", []):
        print(f"  remove {row['family']:<12} {row['fold']} s{row['seed']}: MAE +{row['mae_increase']:.5f}")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
