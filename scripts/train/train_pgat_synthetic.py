"""Gradient-monitored training harness for the synthetic PGAT scenario.

This script orchestrates three stages:
1. Optional regeneration of the synthetic multi-wave dataset.
2. Configuration bootstrap mirroring ``train_dynamic_pgat.py``.
3. A monitored training run that records gradient norms every optimisation step.

Usage
-----
python scripts/train/train_pgat_synthetic.py \
    --config configs/sota_pgat_synthetic.yaml \
    --regenerate-data --rows 4096 --waves 7 --targets 3 --seed 1234
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.data_factory import setup_financial_forecasting_data
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from scripts.train import train_dynamic_pgat

try:
    from scripts.train.generate_synthetic_multi_wave import generate_dataset as _generate_dataset
except ImportError:  # pragma: no cover - guard for refactors
    _generate_dataset = None  # type: ignore[assignment]


@dataclass
class GradientTracker:
    """Collect gradient norms for post-run diagnostics."""

    global_norms: List[float] = field(default_factory=list)
    per_parameter: Dict[str, List[float]] = field(default_factory=dict)
    vanishing_threshold: float = 1e-8
    vanishing_steps: List[int] = field(default_factory=list)

    def log_step(self, named_parameters: Iterable[Tuple[str, torch.nn.Parameter]]) -> None:
        total = 0.0
        step_param_norms: Dict[str, float] = {}
        for name, param in named_parameters:
            if param.grad is None:
                continue
            norm = param.grad.norm(p=2).item()
            step_param_norms[name] = norm
            total += norm * norm
            self.per_parameter.setdefault(name, []).append(norm)
        if total == 0.0:
            self.vanishing_steps.append(len(self.global_norms))
            self.global_norms.append(0.0)
            return
        global_norm = math.sqrt(total)
        if global_norm < self.vanishing_threshold:
            self.vanishing_steps.append(len(self.global_norms))
        self.global_norms.append(global_norm)

    def summary(self) -> Dict[str, float]:
        if not self.global_norms:
            return {"min": 0.0, "max": 0.0, "mean": 0.0}
        return {
            "min": float(min(self.global_norms)),
            "max": float(max(self.global_norms)),
            "mean": float(sum(self.global_norms) / len(self.global_norms)),
        }


class GradientMonitoringExperiment(Exp_Long_Term_Forecast):
    """Specialised experiment that wraps the optimiser for gradient logging."""

    def __init__(self, args: SimpleNamespace, tracker: GradientTracker) -> None:
        self.gradient_tracker = tracker
        super().__init__(args)

    def _select_optimizer(self):  # type: ignore[override]
        optimizer = super()._select_optimizer()
        original_step = optimizer.step
        tracker = self.gradient_tracker
        model = self.model

        def tracked_step(*args, **kwargs):
            tracker.log_step(model.named_parameters())
            return original_step(*args, **kwargs)

        optimizer.step = tracked_step  # type: ignore[assignment]
        return optimizer


def _setup_logging(verbose: bool) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("pgat.synthetic")
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    return log


def _load_config(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    yaml = importlib.import_module("yaml")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_attr(args: SimpleNamespace, name: str, default) -> None:
    if not hasattr(args, name):
        setattr(args, name, default)


def _prepare_args(raw_config: Dict[str, object]) -> SimpleNamespace:
    args = SimpleNamespace(**raw_config)
    args.config = str(raw_config.get("config", ""))
    train_dynamic_pgat._set_common_defaults(args)
    train_dynamic_pgat._configure_pgat_args(args)
    return args


def _auto_device(args: SimpleNamespace, log: logging.Logger) -> None:
    train_dynamic_pgat._auto_select_device(args, log)


def regenerate_dataset_if_needed(args: argparse.Namespace, log: logging.Logger) -> None:
    if not args.regenerate_data:
        return
    if _generate_dataset is None:
        raise RuntimeError("Synthetic generator unavailable; cannot regenerate data.")
    gen_args = argparse.Namespace(
        output=Path(args.dataset_path),
        rows=args.rows,
        waves=args.waves,
        targets=args.targets,
        freq=args.freq,
        start=args.start,
        seed=args.seed,
    )
    log.info(
        "Regenerating synthetic dataset -> %s (rows=%s waves=%s targets=%s)",
        gen_args.output,
        gen_args.rows,
        gen_args.waves,
        gen_args.targets,
    )
    df = _generate_dataset(gen_args)
    gen_args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(gen_args.output, index=False)
    log.info("Synthetic dataset written -> %s", gen_args.output)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sota_pgat_synthetic.yaml"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--regenerate-data", action="store_true")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/synthetic_multi_wave.csv"))
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--waves", type=int, default=7)
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--freq", type=str, default="H")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-threshold", type=float, default=1e-6)
    parser.add_argument("--report", type=Path, default=Path("reports/pgat_synthetic_gradient.json"))
    return parser.parse_args()


def main() -> None:
    cli_args = parse_cli()
    log = _setup_logging(cli_args.verbose)

    regenerate_dataset_if_needed(cli_args, log)

    raw_config = _load_config(cli_args.config)
    raw_config["config"] = str(cli_args.config)
    args = _prepare_args(raw_config)

    # Ensure dataset pointers align with CLI when regen happens
    args.root_path = raw_config.get("root_path", "data")
    args.data_path = raw_config.get("data_path", Path(cli_args.dataset_path).name)
    args.target = raw_config.get("target")

    _ensure_attr(args, "root_path", os.path.join(PROJECT_ROOT, "data"))
    _ensure_attr(args, "data_path", Path(cli_args.dataset_path).name)
    _ensure_attr(args, "target", "target_0,target_1,target_2")

    min_required = getattr(args, "seq_len", 0) + getattr(args, "pred_len", 0)
    _ensure_attr(args, "validation_length", max(150, min_required))
    _ensure_attr(args, "test_length", max(120, min_required))

    _auto_device(args, log)

    dataset_location = Path(str(args.root_path)) / str(args.data_path)
    log.info("Preparing data loaders from %s", dataset_location)
    loaders = setup_financial_forecasting_data(args)
    train_loader, vali_loader, test_loader, scaler_manager, dim_manager = loaders

    args.train_loader = train_loader
    args.vali_loader = vali_loader
    args.test_loader = test_loader
    args.scaler_manager = scaler_manager
    args.dim_manager = dim_manager

    args.enc_in = getattr(args, "enc_in", dim_manager.enc_in)
    args.dec_in = getattr(args, "dec_in", dim_manager.dec_in)
    args.c_out = dim_manager.c_out_model
    args.c_out_evaluation = getattr(args, "c_out_evaluation", dim_manager.c_out_evaluation)

    tracker = GradientTracker(vanishing_threshold=cli_args.gradient_threshold)
    experiment = GradientMonitoringExperiment(args, tracker)

    setting_id = f"synthetic_pgat_{args.features}_{args.seq_len}_{args.pred_len}"
    log.info("Starting training session -> %s", setting_id)
    experiment.train(setting_id)
    log.info("Training finished; launching evaluation")
    experiment.test(setting_id, test=1)

    summary = tracker.summary()
    summary_data: Dict[str, object] = {
        **summary,
        "steps": len(tracker.global_norms),
        "vanishing_steps": tracker.vanishing_steps,
    }

    cli_args.report.parent.mkdir(parents=True, exist_ok=True)
    with cli_args.report.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)

    log.info("Gradient summary -> %s", summary_data)


if __name__ == "__main__":
    main()
