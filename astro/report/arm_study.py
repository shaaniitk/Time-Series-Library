"""Train paired feature arms and issue a pre-registered stop/proceed verdict.

Arms share one channel layout, so a fixed seed gives identical initialization
and data order across real, zero and null arms.  Comparisons are therefore
paired by forecast date, and the only thing that differs between arms is the
content of the known-future block.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

import run
from astro.known import clear_cache

REAL = "real"
ZERO = "zero"
BOOTSTRAP_BLOCK_SESSIONS = 20


def build_args(base_argv: list[str], arm: str, fold: str, seed: int, model_id: str):
    argv = list(base_argv) + [
        "--astro_arm", arm,
        "--astro_fold", fold,
        "--seed", str(seed),
        "--model_id", model_id,
    ]
    clear_cache()
    return run.normalize_args(run.build_parser().parse_args(argv))


def _slug(arm: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in arm)


def predict_eval_block(exp, flag: str = "val", knockout_columns=None) -> pd.DataFrame:
    """Per-date point predictions on an evaluation split.

    ``knockout_columns`` zeroes known-mark columns at inference time: a cheap
    diagnostic on an already-trained model, not a substitute for retraining.
    """

    dataset, loader = exp._get_data(flag)
    if knockout_columns:
        dataset.data_stamp = dataset.data_stamp.copy()
        dataset.data_stamp[:, list(knockout_columns)] = 0.0

    preds, trues = [], []
    exp.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark, positions, valid = (
                exp._unpack_forecast_batch(batch)
            )
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            if positions is not None:
                positions = positions.to(exp.device)
                valid = valid.to(exp.device)
            dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, : exp.args.label_len, :], dec_inp], dim=1)
            output = exp._forward_model(
                batch_x, batch_x_mark, dec_inp.float().to(exp.device), batch_y_mark,
                positions, valid,
            )
            outputs, _, _ = exp._extract_outputs_and_aux(output)
            pred, true = exp._select_targets_for_loss(outputs, batch_y)
            preds.append(pred[:, -1, 0].detach().cpu().numpy())
            trues.append(true[:, -1, 0].detach().cpu().numpy())

    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    offset = int(exp.args.seq_len) + int(exp.args.pred_len) - 1
    dates = dataset.astro_split_dates[offset: offset + len(pred)]
    if len(dates) != len(pred):
        raise RuntimeError("Evaluation predictions do not align with split dates.")
    return pd.DataFrame(
        {"date": dates, "pred": pred, "true": true, "abs_err": np.abs(pred - true)}
    )


@dataclass
class ArmResult:
    arm: str
    fold: str
    seed: int
    frame: pd.DataFrame

    @property
    def mae(self) -> float:
        return float(self.frame["abs_err"].mean())


def train_arm(base_argv, arm, fold, seed, checkpoint_root) -> ArmResult:
    model_id = f"astro_{_slug(arm)}_{fold}_s{seed}"
    args = build_args(
        base_argv + ["--checkpoints", checkpoint_root], arm, fold, seed, model_id
    )
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    exp = Exp_Long_Term_Forecast(args)
    exp.train(model_id)
    return ArmResult(arm, fold, seed, predict_eval_block(exp))


def block_bootstrap_ci(
    deltas: np.ndarray, block: int = BOOTSTRAP_BLOCK_SESSIONS, draws: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% CI of the mean via a moving-block bootstrap (serial dependence aware)."""

    n = len(deltas)
    if n < block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts_max = n - block + 1
    blocks_needed = int(np.ceil(n / block))
    means = np.empty(draws)
    for draw in range(draws):
        starts = rng.integers(0, starts_max, size=blocks_needed)
        sample = np.concatenate([deltas[s: s + block] for s in starts])[:n]
        means[draw] = sample.mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_delta(baseline: ArmResult, candidate: ArmResult) -> np.ndarray:
    """Per-date ``|err_baseline| - |err_candidate|``; positive favors the candidate."""

    merged = baseline.frame.merge(candidate.frame, on="date", suffixes=("_b", "_c"))
    if len(merged) != len(baseline.frame):
        raise RuntimeError("Arms are not evaluated on identical dates.")
    return (merged["abs_err_b"] - merged["abs_err_c"]).to_numpy()


def verdict(results: list[ArmResult], min_winning_folds: int = 3) -> dict:
    """Pre-registered rule: real must beat zero and the best null on enough folds."""

    table: dict[tuple[str, int], dict[str, ArmResult]] = {}
    for result in results:
        table.setdefault((result.fold, result.seed), {})[result.arm] = result

    fold_rows = []
    for (fold, seed), arms in sorted(table.items()):
        if REAL not in arms or ZERO not in arms:
            raise ValueError(f"Fold {fold} seed {seed} needs both real and zero arms.")
        nulls = {name: res for name, res in arms.items() if name.startswith("null:")}
        best_null = min(nulls.values(), key=lambda r: r.mae) if nulls else None
        vs_zero = paired_delta(arms[ZERO], arms[REAL])
        row = {
            "fold": fold,
            "seed": seed,
            "mae_real": arms[REAL].mae,
            "mae_zero": arms[ZERO].mae,
            "delta_vs_zero": float(vs_zero.mean()),
            "ci_vs_zero": block_bootstrap_ci(vs_zero),
            "beats_zero": arms[REAL].mae < arms[ZERO].mae,
        }
        if best_null is not None:
            vs_null = paired_delta(best_null, arms[REAL])
            row.update(
                best_null=best_null.arm,
                mae_best_null=best_null.mae,
                delta_vs_best_null=float(vs_null.mean()),
                ci_vs_best_null=block_bootstrap_ci(vs_null),
                beats_best_null=arms[REAL].mae < best_null.mae,
            )
        else:
            row["beats_best_null"] = False
        fold_rows.append(row)

    by_fold: dict[str, list[dict]] = {}
    for row in fold_rows:
        by_fold.setdefault(row["fold"], []).append(row)
    winning = sorted(
        fold
        for fold, rows in by_fold.items()
        if np.mean([r["beats_zero"] and r["beats_best_null"] for r in rows]) > 0.5
    )
    required = min(min_winning_folds, len(by_fold))
    return {
        "decision": "PROCEED" if len(winning) >= required else "STOP",
        "rule": (
            f"real arm beats both the zero arm and the best null arm on validation MAE "
            f"in at least {required} of {len(by_fold)} folds (majority of seeds per fold)"
        ),
        "winning_folds": winning,
        "rows": fold_rows,
    }


def family_importance(results: list[ArmResult]) -> list[dict]:
    """Leave-one-family-out importance from retrained ``zero:<family>`` arms.

    Positive ``mae_increase`` means removing the family hurt the real arm, i.e.
    the family carried usable information on these dates.
    """

    table: dict[tuple[str, int], dict[str, ArmResult]] = {}
    for result in results:
        table.setdefault((result.fold, result.seed), {})[result.arm] = result
    rows = []
    for (fold, seed), arms in sorted(table.items()):
        if REAL not in arms:
            continue
        for name, result in sorted(arms.items()):
            if not name.startswith(ZERO + ":"):
                continue
            deltas = paired_delta(arms[REAL], result)
            rows.append({
                "fold": fold,
                "seed": seed,
                "family": name.split(":", 1)[1],
                "mae_increase": float(-deltas.mean()),
                "ci": tuple(-bound for bound in reversed(block_bootstrap_ci(deltas))),
            })
    return rows


def write_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
