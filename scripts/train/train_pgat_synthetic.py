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
from typing import Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - resource is unavailable on Windows
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.data_factory import setup_financial_forecasting_data
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from scripts.train import train_dynamic_pgat
from utils.memory_diagnostics import MemoryDiagnostics

try:
    from scripts.train.generate_synthetic_multi_wave import generate_dataset as _generate_dataset
except ImportError:  # pragma: no cover - guard for refactors
    _generate_dataset = None  # type: ignore[assignment]


@dataclass
class ParameterStats:
    """Track streaming statistics for an individual parameter's gradient norms."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = 0.0
    last: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.last = value

    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return self.m2 / (self.count - 1)

    def to_dict(self) -> Dict[str, float]:
        minimum = 0.0 if self.minimum is math.inf else self.minimum
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "min": float(minimum),
            "max": float(self.maximum),
            "last": float(self.last),
            "variance": float(self.variance()),
        }


@dataclass
class GradientTracker:
    """Collect gradient norms and memory metrics for post-run diagnostics."""

    global_norms: List[float] = field(default_factory=list)
    per_parameter: Dict[str, ParameterStats] = field(default_factory=dict)
    vanishing_threshold: float = 1e-8
    vanishing_steps: List[int] = field(default_factory=list)
    rss_history_mb: List[float] = field(default_factory=list)
    cuda_allocated_mb: List[float] = field(default_factory=list)
    max_memory_history: int = 4096

    def log_step(self, named_parameters: Iterable[Tuple[str, torch.nn.Parameter]]) -> None:
        total = 0.0
        for name, param in named_parameters:
            if param.grad is None:
                continue
            norm = param.grad.norm(p=2).item()
            total += norm * norm
            stats = self.per_parameter.setdefault(name, ParameterStats())
            stats.update(norm)
        if total == 0.0:
            self.vanishing_steps.append(len(self.global_norms))
            self.global_norms.append(0.0)
            self._record_memory_stats()
            return
        global_norm = math.sqrt(total)
        if global_norm < self.vanishing_threshold:
            self.vanishing_steps.append(len(self.global_norms))
        self.global_norms.append(global_norm)
        self._record_memory_stats()

    def summary(self) -> Dict[str, float]:
        summary: Dict[str, float] = {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
        if self.global_norms:
            summary.update(
                {
                    "min": float(min(self.global_norms)),
                    "max": float(max(self.global_norms)),
                    "mean": float(sum(self.global_norms) / len(self.global_norms)),
                }
            )
        if self.rss_history_mb:
            summary["rss_mb_max"] = float(max(self.rss_history_mb))
            summary["rss_mb_last"] = float(self.rss_history_mb[-1])
        if self.cuda_allocated_mb:
            summary["cuda_allocated_mb_max"] = float(max(self.cuda_allocated_mb))
            summary["cuda_allocated_mb_last"] = float(self.cuda_allocated_mb[-1])
        return summary

    def parameter_summary(self, limit: int = 15) -> Dict[str, Dict[str, float]]:
        if not self.per_parameter:
            return {}
        ranked = sorted(
            self.per_parameter.items(),
            key=lambda pair: pair[1].maximum,
            reverse=True,
        )[:limit]
        return {name: stats.to_dict() for name, stats in ranked}

    def _record_memory_stats(self) -> None:
        rss_mb = 0.0
        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss = usage.ru_maxrss
            if sys.platform == "darwin":
                rss_mb = rss / (1024.0 * 1024.0)
            else:
                rss_mb = rss / 1024.0
        else:  # pragma: no cover - fallback for platforms without resource
            try:
                import psutil

                process = psutil.Process()
                rss_mb = process.memory_info().rss / (1024.0 ** 2)
            except (ImportError, AttributeError):
                rss_mb = 0.0
        self.rss_history_mb.append(float(rss_mb))
        cuda_mb = 0.0
        if torch.cuda.is_available():
            try:
                cuda_mb = torch.cuda.memory_allocated() / (1024.0 ** 2)
            except RuntimeError:
                cuda_mb = 0.0
        self.cuda_allocated_mb.append(float(cuda_mb))
        if len(self.rss_history_mb) > self.max_memory_history:
            self.rss_history_mb.pop(0)
        if len(self.cuda_allocated_mb) > self.max_memory_history:
            self.cuda_allocated_mb.pop(0)


class GradientMonitoringExperiment(Exp_Long_Term_Forecast):
    """Specialised experiment that wraps the optimiser for gradient logging."""

    def __init__(
        self,
        args: SimpleNamespace,
        tracker: GradientTracker,
        memory_diagnostics: Optional[MemoryDiagnostics] = None,
    ) -> None:
        self.gradient_tracker = tracker
        if memory_diagnostics is not None:
            args.memory_diagnostics = memory_diagnostics
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
    parser.add_argument(
        "--memory-report",
        type=Path,
        default=Path("reports/pgat_synthetic_memory.json"),
        help="Destination file for serialized memory diagnostics snapshots.",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = parse_cli()
    log = _setup_logging(cli_args.verbose)
    memory_diagnostics = MemoryDiagnostics(log)

    memory_diagnostics.snapshot(
        "startup",
        {
            "config": str(cli_args.config),
            "regenerate_data": cli_args.regenerate_data,
            "gradient_threshold": cli_args.gradient_threshold,
        },
    )

    regenerate_dataset_if_needed(cli_args, log)
    if cli_args.regenerate_data:
        memory_diagnostics.snapshot(
            "dataset_regenerated",
            {
                "rows": cli_args.rows,
                "waves": cli_args.waves,
                "targets": cli_args.targets,
            },
        )

    raw_config = _load_config(cli_args.config)
    raw_config["config"] = str(cli_args.config)
    args = _prepare_args(raw_config)
    args.memory_diagnostics = memory_diagnostics

    # Ensure dataset pointers align with CLI when regen happens
    args.root_path = raw_config.get("root_path", "data")
    args.data_path = raw_config.get("data_path", Path(cli_args.dataset_path).name)
    args.target = raw_config.get("target")

    _ensure_attr(args, "root_path", os.path.join(PROJECT_ROOT, "data"))
    _ensure_attr(args, "data_path", Path(cli_args.dataset_path).name)
    _ensure_attr(args, "target", "target_0,target_1,target_2")
    _ensure_attr(args, "checkpoints", os.path.join(PROJECT_ROOT, "checkpoints"))
    Path(str(args.checkpoints)).mkdir(parents=True, exist_ok=True)

    min_required = getattr(args, "seq_len", 0) + getattr(args, "pred_len", 0)
    _ensure_attr(args, "validation_length", max(150, min_required))
    _ensure_attr(args, "test_length", max(120, min_required))

    _auto_device(args, log)

    dataset_location = Path(str(args.root_path)) / str(args.data_path)
    log.info("Preparing data loaders from %s", dataset_location)
    loaders = setup_financial_forecasting_data(args)
    train_loader, vali_loader, test_loader, scaler_manager, dim_manager = loaders

    memory_diagnostics.log_dataloader("train", train_loader)
    memory_diagnostics.log_dataloader("validation", vali_loader)
    memory_diagnostics.log_dataloader("test", test_loader)
    memory_diagnostics.snapshot(
        "dataloaders_built",
        {
            "train_steps": len(train_loader) if hasattr(train_loader, "__len__") else None,
            "validation_steps": len(vali_loader) if hasattr(vali_loader, "__len__") else None,
            "test_steps": len(test_loader) if hasattr(test_loader, "__len__") else None,
        },
    )

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
    experiment = GradientMonitoringExperiment(args, tracker, memory_diagnostics)

    setting_id = f"synthetic_pgat_{args.features}_{args.seq_len}_{args.pred_len}"
    log.info("Starting training session -> %s", setting_id)
    memory_diagnostics.reset_cuda_peaks()
    memory_diagnostics.snapshot(
        "training_start",
        {
            "setting": setting_id,
            "train_epochs": args.train_epochs,
            "batch_size": args.batch_size,
        },
    )
    try:
        experiment.train(setting_id)
        log.info("Training finished; launching evaluation")
        memory_diagnostics.snapshot("training_finished", {"setting": setting_id})
        experiment.test(setting_id, test=1)
        memory_diagnostics.snapshot("evaluation_finished", {"setting": setting_id})

        summary = tracker.summary()
        summary_data: Dict[str, object] = {
            **summary,
            "steps": len(tracker.global_norms),
            "vanishing_steps": tracker.vanishing_steps,
            "parameter_summary_top": tracker.parameter_summary(),
        }

        cli_args.report.parent.mkdir(parents=True, exist_ok=True)
        with cli_args.report.open("w", encoding="utf-8") as handle:
            json.dump(summary_data, handle, indent=2)

        log.info("Gradient summary -> %s", summary_data)
    finally:
        dump_location = memory_diagnostics.dump_history(cli_args.memory_report)
        if dump_location is not None:
            log.info("Memory diagnostics history written -> %s", dump_location)


if __name__ == "__main__":
    main()
