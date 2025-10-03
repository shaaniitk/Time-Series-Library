"""Fixed gradient-monitored training harness for the synthetic PGAT scenario.

This script uses the memory-optimized SOTA_Temporal_PGAT while preserving all
advanced features and the original gradient tracking functionality.

Usage
-----
python scripts/train/train_pgat_synthetic_fixed.py \
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
from scripts.train.memory_efficient_gradient_tracker import MemoryEfficientGradientTracker

# Import the memory-optimized PGAT model
from models.SOTA_Temporal_PGAT_MemoryOptimized import SOTA_Temporal_PGAT_MemoryOptimized

try:
    from scripts.train.generate_synthetic_multi_wave import generate_dataset as _generate_dataset
except ImportError:  # pragma: no cover - guard for refactors
    _generate_dataset = None  # type: ignore[assignment]


class GradientMonitoringExperiment(Exp_Long_Term_Forecast):
    """Specialised experiment that wraps the optimiser for gradient logging."""

    def __init__(
        self,
        args: SimpleNamespace,
        tracker: MemoryEfficientGradientTracker,
        memory_diagnostics: Optional[MemoryDiagnostics] = None,
    ) -> None:
        self.gradient_tracker = tracker
        if memory_diagnostics is not None:
            args.memory_diagnostics = memory_diagnostics
        super().__init__(args)

    def _build_model(self):
        """Override to use enhanced PGAT with all advanced features."""
        if getattr(self.args, 'model', '') in ['SOTA_Temporal_PGAT', 'SOTA_Temporal_PGAT_Enhanced'] or not hasattr(self.args, 'model'):
            # Use enhanced version with all advanced features
            from models.SOTA_Temporal_PGAT_Enhanced import SOTA_Temporal_PGAT_Enhanced
            model = SOTA_Temporal_PGAT_Enhanced(self.args, mode='probabilistic')
            print("✅ Using Enhanced PGAT with advanced features:")
            print(f"   - Mixture Density Networks: {getattr(self.args, 'use_mixture_density', True)}")
            print(f"   - Graph Positional Encoding: {getattr(self.args, 'enable_graph_positional_encoding', True)}")
            print(f"   - Dynamic Edge Weights: {getattr(self.args, 'use_dynamic_edge_weights', True)}")
            print(f"   - Adaptive Temporal Attention: {getattr(self.args, 'use_adaptive_temporal', True)}")
        else:
            # Fallback to parent implementation
            model = super()._build_model()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model

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
    
    # Add memory optimization settings
    _ensure_attr(args, 'use_gradient_checkpointing', True)
    _ensure_attr(args, 'memory_chunk_size', 32)
    _ensure_attr(args, 'enable_memory_cleanup', True)
    _ensure_attr(args, 'use_sparse_adjacency', True)
    
    return args


def _auto_device(args: SimpleNamespace, log: logging.Logger) -> None:
    """Auto-select device with better error handling."""
    try:
        if torch.cuda.is_available():
            args.use_gpu = True
            args.gpu_type = "cuda"
            args.device = torch.device(f'cuda:{args.gpu}')
            log.info("CUDA available - using GPU")
        else:
            args.use_gpu = False
            args.gpu_type = "cpu"
            args.device = torch.device("cpu")
            args.use_amp = False  # Disable mixed precision for CPU
            log.info("CUDA not available - using CPU")
    except Exception as e:
        log.warning(f"GPU detection failed: {e} - falling back to CPU")
        args.use_gpu = False
        args.gpu_type = "cpu"
        args.device = torch.device("cpu")
        args.use_amp = False


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
    parser.add_argument("--batch-size", type=int, default=16, help="Override batch size for memory efficiency")
    parser.add_argument("--disable-amp", action="store_true", help="Disable automatic mixed precision")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_cli()
    log = _setup_logging(cli_args.verbose)
    
    # Use memory-efficient diagnostics
    memory_diagnostics = MemoryDiagnostics(log, keep_last=64, synchronize_cuda=False)

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

    # Override batch size if specified
    if cli_args.batch_size:
        args.batch_size = cli_args.batch_size
        log.info(f"Using batch size: {args.batch_size}")

    # Configure mixed precision (only if CUDA is available)
    if not cli_args.disable_amp and torch.cuda.is_available():
        args.use_amp = True
        log.info("Automatic mixed precision enabled")
    else:
        args.use_amp = False
        if not torch.cuda.is_available():
            log.info("Mixed precision disabled - CUDA not available")

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

    # Use memory-efficient gradient tracker
    tracker = MemoryEfficientGradientTracker(max_global_norms=500)
    experiment = GradientMonitoringExperiment(args, tracker, memory_diagnostics)

    setting_id = f"synthetic_pgat_memory_opt_{args.features}_{args.seq_len}_{args.pred_len}"
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

        # Get memory statistics from the model
        if hasattr(experiment.model, 'get_memory_stats'):
            model_stats = experiment.model.get_memory_stats()
            log.info(f"Model memory stats: {model_stats}")

        summary = {
            "steps": tracker.step_count,
            "vanishing_steps": tracker.vanishing_steps,
            "global_norms_count": len(tracker.global_norms),
            "tracked_parameters": len(tracker.per_parameter),
            "tracker_memory_mb": tracker.get_memory_usage_mb(),
        }

        if len(tracker.global_norms) > 0:
            summary.update({
                "min_gradient_norm": float(min(tracker.global_norms)),
                "max_gradient_norm": float(max(tracker.global_norms)),
                "mean_gradient_norm": float(sum(tracker.global_norms) / len(tracker.global_norms)),
            })

        cli_args.report.parent.mkdir(parents=True, exist_ok=True)
        with cli_args.report.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        log.info("Gradient summary -> %s", summary)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log.error(f"OOM Error despite optimizations: {e}")
            log.error("Try reducing batch size further or disabling some advanced features")
        raise
    finally:
        # Clear model cache
        if hasattr(experiment.model, 'clear_cache'):
            experiment.model.clear_cache()
            
        dump_location = memory_diagnostics.dump_history(cli_args.memory_report)
        if dump_location is not None:
            log.info("Memory diagnostics history written -> %s", dump_location)


if __name__ == "__main__":
    main()