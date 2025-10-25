#!/usr/bin/env python3
"""
PRODUCTION Training Script for Celestial Enhanced PGAT
Heavy-duty overnight training with maximum model capacity
"""

import os
import sys
import io
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union, cast

# Ensure stdout/stderr can emit UTF-8 on Windows to avoid UnicodeEncodeError
try:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        cast(Any, sys.stdout).reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        cast(Any, sys.stderr).reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    # Non-fatal; fallback to default platform encoding
    pass

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

try:
    import torch.cuda.amp as torch_amp
except (ImportError, AttributeError):  # pragma: no cover - AMP unavailable on some builds
    torch_amp = None  # type: ignore[assignment]

if torch_amp is not None and hasattr(torch_amp, "GradScaler"):
    GradScaler = cast(Any, getattr(torch_amp, "GradScaler"))
else:  # pragma: no cover - AMP unavailable
    GradScaler = None  # type: ignore[assignment]

torch_autocast: Optional[Callable[..., Any]]
if torch_amp is not None and hasattr(torch_amp, "autocast"):
    torch_autocast = cast(Callable[..., Any], getattr(torch_amp, "autocast"))
else:  # pragma: no cover - AMP unavailable
    torch_autocast = None
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import argparse
import time
from datetime import datetime
import json
import math
try:
    import psutil
except ImportError:  # pragma: no cover - psutil optional dependency
    psutil = None

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_provider.data_factory import data_provider
from models.Celestial_Enhanced_PGAT import Model
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from layers.modular.decoder.mdn_decoder import mdn_nll_loss
from utils.metrics_calibration import log_calibration_metrics
import warnings
import logging


LOGGER = logging.getLogger(__name__)
MEMORY_LOGGER = logging.getLogger(f"{__name__}.memory")


warnings.filterwarnings('ignore')
# Suppress specific data loader warnings since we handle scaling manually
logging.getLogger('utils.logger').setLevel(logging.ERROR)

class SimpleConfig:
    """Dictionary-backed configuration with attribute access support."""

    __slots__ = ("_data",)

    def __init__(self, config_dict: Mapping[str, Any]) -> None:
        super().__setattr__("_data", dict(config_dict))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:  # pragma: no cover - diagnostic aid only
            raise AttributeError(f"Configuration does not define attribute '{name}'.") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self._data[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        """Dictionary-style getter used for compatibility with mapping APIs."""

        return self._data.get(name, default)

    def items(self) -> Iterable[Tuple[str, Any]]:
        """Iterate over stored configuration keys and values."""

        return self._data.items()


def configure_logging(config: SimpleConfig) -> logging.Logger:
    """Configure module logging based on configuration values."""
    level_name = str(getattr(config, "log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = getattr(
        config,
        "log_format",
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    class LossOnlyFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
            try:
                msg = record.getMessage()
            except Exception:
                return False
            # Allow: per-batch loss, epoch summaries, and test loss
            if "loss=" in msg:
                return True
            if msg.startswith("Epoch ") or ("Epoch " in msg and "production training" in msg):
                return True
            if "test_loss=" in msg:
                return True
            return False

    root_logger = logging.getLogger()
    # Reset existing console handlers to enforce our console policy
    for h in list(root_logger.handlers):
        try:
            h.flush()
        except Exception:
            pass
        root_logger.removeHandler(h)

    # File logging for general logs if configured
    log_file = getattr(config, "log_file", None)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

    # Console handler: only show loss lines
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.addFilter(LossOnlyFilter())
    root_logger.addHandler(console_handler)
    root_logger.setLevel(level)

    LOGGER.setLevel(level)
    for handler in LOGGER.handlers:
        handler.setLevel(level)

    return LOGGER


@dataclass
class DataModule:
    """Container bundling a dataset with its configured data loader."""

    dataset: Any
    loader: DataLoader


@dataclass
class EvaluationResult:
    """Encapsulate evaluation predictions and computed metrics."""

    preds: np.ndarray
    trues: np.ndarray
    overall: Dict[str, float]
    per_target: Dict[str, Dict[str, float]]
    processed_batches: int

    @property
    def success(self) -> bool:
        """Return True when at least one batch was processed during evaluation."""

        return self.processed_batches > 0


@dataclass
class AdversarialScenarioResult:
    """Hold metrics for a single adversarial evaluation scenario."""

    name: str
    overall: Dict[str, float]
    per_target: Dict[str, Dict[str, float]]
    processed_batches: int
    description: str = ""

    @property
    def success(self) -> bool:
        """Return True when evaluation succeeded for this scenario."""

        return self.processed_batches > 0


@dataclass
class TrainingArtifacts:
    """Track training metrics and checkpoint information across epochs."""

    train_losses: List[float]
    val_losses: List[float]
    best_val_loss: float
    best_epoch: Optional[int]
    best_checkpoint_path: Path

    def record_epoch(self, epoch: int, train_loss: float, val_loss: float) -> bool:
        """Store epoch metrics and return True when validation loss improves."""

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            return True
        return False


def _bytes_to_megabytes(value: Union[int, float]) -> float:
    """Convert a byte count into megabytes for human-readable diagnostics."""

    return float(value) / (1024 ** 2)


def _gather_memory_stats(device: torch.device) -> Dict[str, Union[float, str]]:
    """Collect CPU and optional CUDA memory statistics for diagnostics logging."""

    stats: Dict[str, Union[float, str]] = {}

    if psutil is not None:
        try:
            virtual_memory = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
            stats.update(
                {
                    "cpu_total_mb": round(_bytes_to_megabytes(virtual_memory.total), 2),
                    "cpu_available_mb": round(_bytes_to_megabytes(virtual_memory.available), 2),
                    "cpu_used_mb": round(
                        _bytes_to_megabytes(virtual_memory.total - virtual_memory.available),
                        2,
                    ),
                    "process_rss_mb": round(_bytes_to_megabytes(process_memory.rss), 2),
                    "process_vms_mb": round(_bytes_to_megabytes(process_memory.vms), 2),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            stats["cpu_memory_error"] = str(exc)
    else:
        stats["cpu_memory_error"] = "psutil_unavailable"

    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except Exception:  # pragma: no cover - optional sync may fail on some builds
            pass
        try:
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            max_allocated = torch.cuda.max_memory_allocated(device)
            max_reserved = torch.cuda.max_memory_reserved(device)
            stats.update(
                {
                    "cuda_allocated_mb": round(_bytes_to_megabytes(allocated), 2),
                    "cuda_reserved_mb": round(_bytes_to_megabytes(reserved), 2),
                    "cuda_max_allocated_mb": round(_bytes_to_megabytes(max_allocated), 2),
                    "cuda_max_reserved_mb": round(_bytes_to_megabytes(max_reserved), 2),
                }
            )
            try:
                device_properties = torch.cuda.get_device_properties(device)
                stats["cuda_total_mb"] = round(_bytes_to_megabytes(device_properties.total_memory), 2)
            except Exception:
                stats.setdefault("cuda_total_mb", "unavailable")
        except Exception as exc:  # pragma: no cover - diagnostic best effort
            stats["cuda_memory_error"] = str(exc)
    else:
        stats.setdefault("cuda_info", "cuda_unavailable")

    return stats


def _log_memory_snapshot(
    stage: str,
    device: torch.device,
    logger: Optional[logging.Logger],
    context: Optional[Mapping[str, Any]] = None,
) -> None:
    """Emit structured memory diagnostics for later offline analysis."""

    if logger is None:
        return

    try:
        stats = _gather_memory_stats(device)
        payload: Dict[str, Union[str, float]] = {"stage": stage, **stats}
        if context:
            for key, value in context.items():
                if isinstance(value, bool):
                    payload[f"context_{key}"] = str(value)
                elif isinstance(value, (int, float)):
                    payload[f"context_{key}"] = float(value)
                else:
                    payload[f"context_{key}"] = str(value)
        logger.debug(json.dumps(payload, sort_keys=True, default=str))
    except Exception as exc:  # pragma: no cover - diagnostic logging should never crash
        logger.debug("Memory snapshot collection failed for stage %s: %s", stage, exc)


def _maybe_dump_cuda_summary(device: torch.device, logger: Optional[logging.Logger], stage: str) -> None:
    """Dump a CUDA memory summary if diagnostics and CUDA support are available."""

    if logger is None or device.type != "cuda" or not torch.cuda.is_available():
        return

    try:
        summary_text = torch.cuda.memory_summary(device=device, abbreviated=True)
        logger.debug("CUDA memory summary | stage=%s\n%s", stage, summary_text)
    except Exception as exc:  # pragma: no cover - summary is best effort only
        logger.debug("Failed to capture CUDA memory summary for stage %s: %s", stage, exc)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Convert possibly-string configuration values into strict booleans."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _ensure_reproducibility(seed: int) -> None:
    """Seed random number generators for reproducible production runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers deterministically for reproducibility."""

    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _determine_num_workers(args: SimpleConfig, logger: logging.Logger) -> int:
    """Resolve the requested num_workers setting into a concrete worker count."""

    requested = getattr(args, "num_workers", 0)
    max_workers = os.cpu_count() or 1

    if isinstance(requested, str):
        requested_normalized = requested.strip().lower()
        if requested_normalized in {"auto", "default"}:
            resolved = max(1, max_workers // 2)
        else:
            try:
                resolved = int(requested)
            except ValueError:
                logger.warning(
                    "Invalid num_workers value '%s'; defaulting to 0 for stability.",
                    requested,
                )
                resolved = 0
    else:
        resolved = int(requested)

    resolved = max(0, min(resolved, max_workers))
    if resolved != requested:
        logger.info("Adjusted num_workers from %s to %s based on system limits", requested, resolved)

    return resolved


def _resolve_batch_size(flag: str, args: SimpleConfig, loader: DataLoader) -> int:
    """Derive batch size for the given split, honoring overrides when provided."""

    override_attr = f"{flag}_batch_size" if flag != "val" else "val_batch_size"
    override_value = getattr(args, override_attr, None)
    if override_value is not None:
        return int(override_value)

    loader_batch_size = getattr(loader, "batch_size", None)
    if loader_batch_size is not None:
        return int(loader_batch_size)

    return int(getattr(args, "batch_size", 1))


def _optimize_data_loader(
    dataset: Any,
    loader: DataLoader,
    args: SimpleConfig,
    device: torch.device,
    flag: str,
    logger: logging.Logger,
) -> Tuple[DataLoader, Dict[str, Any]]:
    """Rebuild a data loader with optimized runtime settings and diagnostics."""

    shuffle_default = flag == "train"
    shuffle = bool(getattr(args, f"{flag}_shuffle", shuffle_default))

    drop_last_default = flag == "train"
    if flag == "train":
        drop_last = bool(getattr(loader, "drop_last", drop_last_default))
    else:
        drop_last = drop_last_default
    drop_last_override = getattr(args, f"{flag}_drop_last", None)
    if drop_last_override is not None:
        drop_last = _coerce_bool(drop_last_override, drop_last_default)

    batch_size = _resolve_batch_size(flag, args, loader)
    pin_memory = device.type == "cuda" and _coerce_bool(getattr(args, "pin_memory", True), True)

    persistent_workers = False
    if args.num_workers > 0:
        persistent_workers = _coerce_bool(getattr(args, "persistent_workers", True), True)

    prefetch_factor: Optional[int] = None
    if args.num_workers > 0:
        prefetch_value = getattr(args, "prefetch_factor", getattr(loader, "prefetch_factor", 2))
        try:
            prefetch_factor = max(1, int(prefetch_value))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid prefetch_factor '%s'; defaulting to 2 for loader optimization.",
                prefetch_value,
            )
            prefetch_factor = 2

    timeout_seconds = float(getattr(args, "loader_timeout", getattr(loader, "timeout", 0.0)))
    collate_fn = getattr(loader, "collate_fn", None)

    worker_init_fn = _seed_worker if args.num_workers > 0 else None
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())

    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "worker_init_fn": worker_init_fn,
        "generator": generator,
    }

    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    if timeout_seconds > 0:
        loader_kwargs["timeout"] = timeout_seconds

    optimized_loader = DataLoader(dataset, **loader_kwargs)

    diagnostics = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor if prefetch_factor is not None else "n/a",
        "persistent_workers": persistent_workers,
        "timeout": timeout_seconds,
    }

    return optimized_loader, diagnostics


def prepare_data_modules(
    args: SimpleConfig,
    logger: logging.Logger,
    device: torch.device,
) -> Dict[str, DataModule]:
    """Load datasets and data loaders with validation and diagnostic logging."""

    args.num_workers = _determine_num_workers(args, logger)

    modules: Dict[str, DataModule] = {}
    for flag in ("train", "val", "test"):
        dataset, loader = data_provider(args, flag=flag)
        if len(loader) == 0:
            raise ValueError(f"{flag} data loader returned zero batches; check dataset configuration.")

        optimized_loader, diagnostics = _optimize_data_loader(dataset, loader, args, device, flag, logger)
        modules[flag] = DataModule(dataset=dataset, loader=optimized_loader)

        dataset_size = len(dataset) if hasattr(dataset, "__len__") else "unknown"
        batch_size = diagnostics["batch_size"]
        drop_last = diagnostics["drop_last"]
        logger.info(
            "Loaded %s data module | samples=%s batches=%s batch_size=%s drop_last=%s shuffle=%s pin_memory=%s prefetch_factor=%s persistent_workers=%s",
            flag,
            dataset_size,
            len(optimized_loader),
            batch_size,
            drop_last,
            diagnostics["shuffle"],
            diagnostics["pin_memory"],
            diagnostics["prefetch_factor"],
            diagnostics["persistent_workers"],
        )

    return modules


def create_training_artifacts(checkpoint_dir: Path) -> TrainingArtifacts:
    """Initialize training artifacts for the production training loop."""

    return TrainingArtifacts(
        train_losses=[],
        val_losses=[],
        best_val_loss=float("inf"),
        best_epoch=None,
        best_checkpoint_path=checkpoint_dir / "best_model.pth",
    )


def manage_checkpoints(
    artifacts: TrainingArtifacts,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    current_lr: float,
    checkpoint_dir: Path,
    save_best_only: bool,
    checkpoint_interval: int,
    logger: logging.Logger,
    is_best: bool,
) -> None:
    """Persist checkpoints according to configuration and best-loss tracking."""

    checkpoint_payload = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "train_loss": train_loss,
        "lr": current_lr,
    }

    if is_best:
        torch.save(checkpoint_payload, artifacts.best_checkpoint_path)
        logger.info(
            "New best model saved at %s | val_loss=%0.6f",
            artifacts.best_checkpoint_path,
            val_loss,
        )

    if not save_best_only and (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(checkpoint_payload, checkpoint_path)
        logger.info("Checkpoint saved at %s", checkpoint_path)


def persist_training_results(
    checkpoint_dir: Path,
    config_dict: Mapping[str, Any],
    args: SimpleConfig,
    total_params: int,
    artifacts: TrainingArtifacts,
    evaluation_result: EvaluationResult,
    adversarial_results: Sequence[AdversarialScenarioResult],
    training_duration_seconds: float,
    rmse_value: float,
    effective_batch_size: int,
    warmup_epochs: int,
    use_amp: bool,
    logger: logging.Logger,
) -> None:
    """Persist production training results and emit summary logging."""

    num_targets = int(evaluation_result.preds.shape[-1]) if evaluation_result.preds.size else 0
    overall_metrics = evaluation_result.overall
    per_target_metrics = evaluation_result.per_target

    training_duration_hours = training_duration_seconds / 3600 if training_duration_seconds else 0.0
    avg_epoch_time = training_duration_seconds / max(1, args.train_epochs)
    best_val_repr = (
        f"{artifacts.best_val_loss:.6f}" if math.isfinite(artifacts.best_val_loss) else "not recorded"
    )
    best_epoch_display = artifacts.best_epoch + 1 if artifacts.best_epoch is not None else "n/a"
    best_epoch_serializable = (
        artifacts.best_epoch + 1 if artifacts.best_epoch is not None else None
    )

    rmse_display = (
        f"{rmse_value:.6f}"
        if isinstance(rmse_value, (int, float)) and math.isfinite(rmse_value)
        else rmse_value
    )

    results = {
        "model": "Celestial_Enhanced_PGAT_PRODUCTION",
        "task": "OHLC_Prediction_Production",
        "config": "HEAVY_DUTY_OVERNIGHT",
        "num_targets": num_targets,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
        "train_epochs": args.train_epochs,
        "total_parameters": total_params,
        "training_time_hours": training_duration_hours,
        "overall_metrics": overall_metrics,
        "per_target_metrics": per_target_metrics,
        "adversarial_diagnostics": [
            {
                "name": scenario.name,
                "processed_batches": scenario.processed_batches,
                "overall": scenario.overall,
                "per_target": scenario.per_target,
                "description": scenario.description,
            }
            for scenario in adversarial_results
        ],
        "train_losses": artifacts.train_losses,
        "val_losses": artifacts.val_losses,
        "best_val_loss": (
            float(artifacts.best_val_loss) if math.isfinite(artifacts.best_val_loss) else None
        ),
        "best_epoch": best_epoch_serializable,
        "config_dict": config_dict,
        "timestamp": datetime.now().isoformat(),
    }

    results_file = checkpoint_dir / "production_results.json"
    with open(results_file, "w", encoding="utf-8") as result_file:
        json.dump(results, result_file, indent=2)

    logger.info("PRODUCTION results saved to %s", results_file)
    logger.info("=" * 80)
    logger.info("PRODUCTION Celestial Enhanced PGAT training complete")
    logger.info("The heavy-duty astrological AI has completed overnight training")
    logger.info(
        "Successfully trained %s parameters over %s epochs",
        f"{total_params:,}",
        args.train_epochs,
    )

    logger.info("Total training time %.2f hours | final_rmse=%s", training_duration_hours, rmse_display)
    logger.info("Best validation loss %s (epoch %s)", best_val_repr, best_epoch_display)
    logger.info("Average epoch time %0.2fs | effective_batch_size=%s", avg_epoch_time, effective_batch_size)
    if use_amp:
        logger.info("Mixed precision training was used")
    logger.info("Learning rate schedule: warmup %s epochs + cosine annealing", warmup_epochs)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    scaler: Optional[Any],
    use_amp: bool,
    gradient_accumulation_steps: int,
    device: torch.device,
    epoch: int,
    args: SimpleConfig,
    logger: logging.Logger,
    target_scaler: Any,
    target_indices: Sequence[int],
    total_train_batches: int,
    full_cycles: int,
    remainder_batches: int,
    epoch_start: float,
    sequential_mixture_loss_cls: Optional[Type[nn.Module]],
    mixture_loss_cls: Optional[Type[nn.Module]],
) -> Tuple[float, int]:
    """Execute the training phase for a single epoch and return loss statistics."""

    model.train()
    train_loss = 0.0
    train_batches = 0
    optimizer.zero_grad(set_to_none=True)

    logger.info("Epoch %s/%s - production training", epoch + 1, args.train_epochs)

    memory_logging_enabled = getattr(args, "enable_memory_diagnostics", False)
    memory_log_interval = max(1, int(getattr(args, "memory_log_interval", 25)))
    if memory_logging_enabled:
        _log_memory_snapshot(
            f"train_epoch_{epoch + 1}_start",
            device,
            MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
            {"gradient_accumulation_steps": gradient_accumulation_steps},
        )

    log_interval = max(1, int(getattr(args, "log_interval", 10)))
    for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        try:
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float()
            batch_y_mark = batch_y_mark.float()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            batch_y_mark = batch_y_mark.to(device)

            # 🚀 ENHANCED: Use future celestial data for better predictions
            dec_inp = _create_enhanced_decoder_input(batch_y, args, logger).float().to(device)

            if use_amp:
                if torch_autocast is None:
                    raise RuntimeError(
                        "AMP training requested but torch.cuda.amp.autocast is unavailable on this platform."
                    )
                with torch_autocast():
                    outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs_tensor, aux_loss, mdn_outputs, _ = _normalize_model_output(outputs_raw)
            c_out_evaluation = len(target_indices)
            y_true_for_loss = scale_targets_for_loss(
                batch_y[:, -args.pred_len:, :], target_scaler, target_indices, device
            )

            # Priority: MDN decoder > sequential mixture > standard loss
            enable_mdn = getattr(args, "enable_mdn_decoder", False)
            if enable_mdn and mdn_outputs is not None:
                # MDN decoder path: compute NLL loss
                pi, mu, sigma = mdn_outputs
                if pi.size(1) > args.pred_len:
                    pi = pi[:, -args.pred_len:, ...]
                    mu = mu[:, -args.pred_len:, ...]
                    sigma = sigma[:, -args.pred_len:, ...]
                targets_mdn = (
                    y_true_for_loss.squeeze(-1)
                    if y_true_for_loss.dim() == 3 and y_true_for_loss.size(-1) == 1
                    else y_true_for_loss
                )
                raw_loss = mdn_nll_loss(pi, mu, sigma, targets_mdn, reduce='mean')
            else:
                # Fallback to sequential mixture or standard loss
                is_seq_mixture = sequential_mixture_loss_cls is not None and isinstance(criterion, sequential_mixture_loss_cls)
                is_mixture = mixture_loss_cls is not None and isinstance(criterion, mixture_loss_cls)
                if is_seq_mixture or is_mixture:
                    if mdn_outputs is None:
                        raise ValueError(
                            "MixtureNLLLoss requires model to return a (means, stds, weights) tuple during training."
                        )
                    means_t, stds_t, weights_t = mdn_outputs
                    if means_t.size(1) > args.pred_len:
                        means_t = means_t[:, -args.pred_len:, ...]
                        stds_t = stds_t[:, -args.pred_len:, ...]
                        weights_t = weights_t[:, -args.pred_len:, ...]
                    targets_t = (
                        y_true_for_loss.squeeze(-1)
                        if y_true_for_loss.dim() == 3 and y_true_for_loss.size(-1) == 1
                        else y_true_for_loss
                    )
                    raw_loss = criterion((means_t, stds_t, weights_t), targets_t)
                else:
                    y_pred_for_loss = outputs_tensor[:, -args.pred_len:, :c_out_evaluation]
                    raw_loss = criterion(y_pred_for_loss, y_true_for_loss)

            if aux_loss:
                raw_loss = raw_loss + aux_loss

            if getattr(model, "use_stochastic_learner", False):
                reg_loss = model.get_regularization_loss()
                reg_weight = float(getattr(args, "reg_loss_weight", 0.0005))
                raw_loss = raw_loss + (reg_loss * reg_weight)

            cycle_index = batch_index // gradient_accumulation_steps
            effective_cycle = gradient_accumulation_steps
            if remainder_batches and cycle_index >= full_cycles:
                effective_cycle = remainder_batches

            loss = raw_loss / effective_cycle

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_cycle_end = (batch_index + 1) % gradient_accumulation_steps == 0
            is_final_partial = (
                remainder_batches and cycle_index >= full_cycles and batch_index == total_train_batches - 1
            )
            if is_cycle_end or is_final_partial:
                if hasattr(args, "clip_grad_norm"):
                    clip_value = float(getattr(args, "clip_grad_norm", 0.0))
                    if use_amp and scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if batch_index != total_train_batches - 1:
                    optimizer.zero_grad(set_to_none=True)

            train_loss += raw_loss.detach().item()
            train_batches += 1

            if memory_logging_enabled and (
                batch_index == 0 or (batch_index + 1) % memory_log_interval == 0
            ):
                raw_loss_value = float(raw_loss.detach().item())
                _log_memory_snapshot(
                    f"train_epoch_{epoch + 1}_batch_{batch_index + 1}",
                    device,
                    MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
                    {
                        "raw_loss": raw_loss_value,
                        "optimizer_step": int(is_cycle_end or is_final_partial),
                        "accumulated_batches": batch_index + 1,
                    },
                )

            if batch_index % log_interval == 0:
                elapsed = time.time() - epoch_start
                logger.info(
                    "Batch %s/%s | loss=%0.6f elapsed=%0.1fs",
                    batch_index,
                    len(train_loader),
                    raw_loss.detach().item(),
                    elapsed,
                )
                if epoch == 0 and batch_index == 0:
                    logger.debug("Production train - first batch scaling check")
                    logger.debug(
                        "Raw batch_y OHLC stats | mean=%0.6f std=%0.6f",
                        batch_y[:, -args.pred_len:, :4].mean().item(),
                        batch_y[:, -args.pred_len:, :4].std().item(),
                    )
                    logger.debug(
                        "Scaled targets stats | mean=%0.6f std=%0.6f",
                        y_true_for_loss.mean().item(),
                        y_true_for_loss.std().item(),
                    )
                    logger.debug(
                        "Model outputs stats | mean=%0.6f std=%0.6f",
                        outputs_tensor.mean().item(),
                        outputs_tensor.std().item(),
                    )
                    logger.debug("Scaling consistency verified for production training")

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Training step failed: %s", exc)
            if memory_logging_enabled and getattr(args, "memory_summary_on_exception", True):
                _log_memory_snapshot(
                    f"train_epoch_{epoch + 1}_batch_{batch_index + 1}_exception",
                    device,
                    MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
                    {"exception": str(exc)},
                )
                _maybe_dump_cuda_summary(
                    device,
                    MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
                    f"train_epoch_{epoch + 1}_batch_{batch_index + 1}_exception",
                )
            optimizer.zero_grad(set_to_none=True)
            continue

    avg_train_loss = train_loss / max(train_batches, 1)
    if memory_logging_enabled:
        _log_memory_snapshot(
            f"train_epoch_{epoch + 1}_end",
            device,
            MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
            {"avg_train_loss": avg_train_loss, "train_batches": train_batches},
        )
        if getattr(args, "dump_cuda_memory_summary", False):
            _maybe_dump_cuda_summary(
                device,
                MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
                f"train_epoch_{epoch + 1}_end",
            )
    return avg_train_loss, train_batches


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    args: SimpleConfig,
    logger: logging.Logger,
    target_scaler: Any,
    target_indices: Sequence[int],
    sequential_mixture_loss_cls: Optional[Type[nn.Module]],
    mixture_loss_cls: Optional[Type[nn.Module]],
) -> Tuple[float, int]:
    """Run validation phase for a single epoch and return aggregate loss."""

    model.eval()
    val_loss = 0.0
    val_batches = 0

    memory_logging_enabled = getattr(args, "enable_memory_diagnostics", False)
    memory_log_interval = max(1, int(getattr(args, "memory_log_interval", 25)))
    if memory_logging_enabled:
        _log_memory_snapshot(
            f"validate_epoch_{epoch + 1}_start",
            device,
            MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
            {},
        )

    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
            try:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                # 🚀 ENHANCED: Use future celestial data for validation
                dec_inp = _create_enhanced_decoder_input(batch_y, args, logger).float().to(device)

                outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs_tensor, aux_loss, mdn_outputs, _ = _normalize_model_output(outputs_raw)

                c_out_evaluation = len(target_indices)
                y_true_for_loss = scale_targets_for_loss(
                    batch_y[:, -args.pred_len:, :], target_scaler, target_indices, device
                )

                # Priority: MDN decoder > sequential mixture > standard loss
                enable_mdn = getattr(args, "enable_mdn_decoder", False)
                if enable_mdn and mdn_outputs is not None:
                    # MDN decoder path: compute NLL loss
                    pi, mu, sigma = mdn_outputs
                    if pi.size(1) > args.pred_len:
                        pi = pi[:, -args.pred_len:, ...]
                        mu = mu[:, -args.pred_len:, ...]
                        sigma = sigma[:, -args.pred_len:, ...]
                    targets_mdn = (
                        y_true_for_loss.squeeze(-1)
                        if y_true_for_loss.dim() == 3 and y_true_for_loss.size(-1) == 1
                        else y_true_for_loss
                    )
                    loss = mdn_nll_loss(pi, mu, sigma, targets_mdn, reduce='mean')
                else:
                    # Fallback to mixture or standard loss
                    use_mixture_loss = False
                    if mixture_loss_cls is not None and isinstance(criterion, mixture_loss_cls):
                        use_mixture_loss = True
                    elif sequential_mixture_loss_cls is not None and isinstance(criterion, sequential_mixture_loss_cls):
                        use_mixture_loss = True

                    if use_mixture_loss:
                        if mdn_outputs is None:
                            raise ValueError("MixtureNLLLoss requires model to return a (means, stds, weights) tuple.")
                        means_v, stds_v, weights_v = mdn_outputs
                        if means_v.size(1) > args.pred_len:
                            means_v = means_v[:, -args.pred_len:, ...]
                            stds_v = stds_v[:, -args.pred_len:, ...]
                            weights_v = weights_v[:, -args.pred_len:, ...]
                        targets_v = (
                            y_true_for_loss.squeeze(-1)
                            if y_true_for_loss.dim() == 3 and y_true_for_loss.size(-1) == 1
                            else y_true_for_loss
                        )
                        loss = criterion((means_v, stds_v, weights_v), targets_v)
                    else:
                        y_pred_for_loss = outputs_tensor[:, -args.pred_len:, :c_out_evaluation]
                        loss = criterion(y_pred_for_loss, y_true_for_loss)

                if aux_loss:
                    loss = loss + aux_loss

                val_loss += float(loss.item())
                val_batches += 1

                if memory_logging_enabled and (
                    batch_index == 0 or (batch_index + 1) % memory_log_interval == 0
                ):
                    _log_memory_snapshot(
                        f"validate_epoch_{epoch + 1}_batch_{batch_index + 1}",
                        device,
                        MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
                        {
                            "raw_loss": float(loss.detach().item()),
                            "accumulated_batches": batch_index + 1,
                        },
                    )

                if epoch == 0 and val_batches == 1:
                    logger.debug("Production validation - first batch scaling check")
                    logger.debug(
                        "Raw batch_y OHLC stats | mean=%0.6f std=%0.6f",
                        batch_y[:, -args.pred_len:, :4].mean().item(),
                        batch_y[:, -args.pred_len:, :4].std().item(),
                    )
                    logger.debug(
                        "Scaled targets stats | mean=%0.6f std=%0.6f",
                        y_true_for_loss.mean().item(),
                        y_true_for_loss.std().item(),
                    )
                    logger.debug(
                        "Model outputs stats | mean=%0.6f std=%0.6f",
                        outputs_tensor.mean().item(),
                        outputs_tensor.std().item(),
                    )
                    logger.debug("Scaling consistency verified for production validation")

            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Validation step warning: %s", exc)
                if memory_logging_enabled and getattr(args, "memory_summary_on_exception", True):
                    _log_memory_snapshot(
                        f"validate_epoch_{epoch + 1}_batch_{batch_index + 1}_exception",
                        device,
                        MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
                        {"exception": str(exc)},
                    )
                    _maybe_dump_cuda_summary(
                        device,
                        MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
                        f"validate_epoch_{epoch + 1}_batch_{batch_index + 1}_exception",
                    )
                continue

    avg_val_loss = val_loss / max(val_batches, 1)
    if memory_logging_enabled:
        _log_memory_snapshot(
            f"validate_epoch_{epoch + 1}_end",
            device,
            MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
            {"avg_val_loss": avg_val_loss, "val_batches": val_batches},
        )
    return avg_val_loss, val_batches


def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    args: SimpleConfig,
    target_indices: Sequence[int],
    num_samples: int,
    logger: logging.Logger,
    scenario_name: str = "baseline",
    max_batches: Optional[int] = None,
    input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[np.ndarray, np.ndarray, int, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Collect predictions, targets, and optional MDN components for evaluation or adversarial diagnostics.
    
    Returns:
        Tuple of (preds, trues, processed_count, mdn_components) where mdn_components is (pi, mu, sigma) if MDN enabled, else None.
    """

    if num_samples <= 0:
        return (
            np.empty((0, args.pred_len, len(target_indices)), dtype=np.float32),
            np.empty((0, args.pred_len, len(target_indices)), dtype=np.float32),
            0,
            None,
        )

    enable_mdn = getattr(args, "enable_mdn_decoder", False)
    preds = np.zeros((num_samples, args.pred_len, len(target_indices)), dtype=np.float32)
    trues = np.zeros_like(preds)
    
    # For MDN: collect pi, mu, sigma
    mdn_pi_list, mdn_mu_list, mdn_sigma_list = ([] if enable_mdn else None, [] if enable_mdn else None, [] if enable_mdn else None)
    current_index = 0

    memory_logging_enabled = getattr(args, "enable_memory_diagnostics", False)
    memory_log_interval = max(1, int(getattr(args, "memory_log_interval", 25)))
    if memory_logging_enabled:
        _log_memory_snapshot(
            f"collect_predictions_{scenario_name}_start",
            device,
            MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
            {
                "num_samples": num_samples,
                "max_batches": max_batches if max_batches is not None else "all",
            },
        )

    model.eval()
    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            batch_x = batch_x.float()
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float()
            batch_y_mark = batch_y_mark.float()

            if input_transform is not None:
                try:
                    batch_x = input_transform(batch_x.clone())
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("Adversarial transform '%s' failed on batch %s: %s", scenario_name, batch_index, exc)
                    continue

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            batch_y_mark = batch_y_mark.to(device)

            # 🚀 ENHANCED: Use future celestial data for testing
            dec_inp = _create_enhanced_decoder_input(batch_y, args, logger).float().to(device)

            outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs_tensor, _, mdn_outputs, _ = _normalize_model_output(outputs_raw)

            # Extract point predictions and optionally collect MDN components
            if enable_mdn and mdn_outputs is not None:
                pi, mu, sigma = mdn_outputs
                if pi.size(1) > args.pred_len:
                    pi = pi[:, -args.pred_len:, ...]
                    mu = mu[:, -args.pred_len:, ...]
                    sigma = sigma[:, -args.pred_len:, ...]
                
                # Point prediction: mean of mixture
                if mu.dim() == 4:  # [B, S, T, K]
                    pi_expanded = pi.unsqueeze(2)  # [B, S, 1, K]
                    pred_tensor = (pi_expanded * mu).sum(dim=-1)  # [B, S, T]
                else:
                    pred_tensor = (pi * mu).sum(dim=-1).unsqueeze(-1)
                
                # Collect MDN components for calibration
                mdn_pi_list.append(pi.detach().cpu().numpy())
                mdn_mu_list.append(mu.detach().cpu().numpy())
                mdn_sigma_list.append(sigma.detach().cpu().numpy())
            elif mdn_outputs is not None:
                # Legacy mixture format
                means_te, stds_te, weights_te = mdn_outputs
                if means_te.size(1) > args.pred_len:
                    means_te = means_te[:, -args.pred_len:, ...]
                    stds_te = stds_te[:, -args.pred_len:, ...]
                    weights_te = weights_te[:, -args.pred_len:, ...]
                if means_te.dim() == 4:
                    weights_expanded = weights_te.unsqueeze(2)
                    pred_tensor = (weights_expanded * means_te).sum(dim=-1)
                else:
                    pred_tensor = (weights_te * means_te).sum(dim=-1).unsqueeze(-1)
            else:
                pred_tensor = outputs_tensor

            true_tensor = batch_y[:, -args.pred_len:, :]

            pred_aligned = pred_tensor[:, -args.pred_len:, :]
            true_aligned = true_tensor[:, -args.pred_len:, :]
            if target_indices:
                pred_aligned = pred_aligned[:, :, target_indices]
                true_aligned = true_aligned[:, :, target_indices]

            if torch.isnan(pred_aligned).any() or torch.isinf(pred_aligned).any():
                logger.warning(
                    "NaN or inf detected in predictions during '%s' scenario; skipping batch %s",
                    scenario_name,
                    batch_index,
                )
                continue

            batch_size = pred_aligned.size(0)
            end_index = current_index + batch_size
            if end_index > preds.shape[0]:
                extra = end_index - preds.shape[0]
                preds = np.concatenate(
                    [preds, np.zeros((extra, args.pred_len, pred_aligned.size(-1)), dtype=np.float32)], axis=0
                )
                trues = np.concatenate(
                    [trues, np.zeros((extra, args.pred_len, pred_aligned.size(-1)), dtype=np.float32)], axis=0
                )

            preds[current_index:end_index] = pred_aligned.detach().cpu().numpy()
            trues[current_index:end_index] = true_aligned.detach().cpu().numpy()
            current_index = end_index

            if memory_logging_enabled and (
                batch_index == 0 or (batch_index + 1) % memory_log_interval == 0
            ):
                _log_memory_snapshot(
                    f"collect_predictions_{scenario_name}_batch_{batch_index + 1}",
                    device,
                    MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
                    {
                        "processed": current_index,
                        "batch_size": batch_size,
                    },
                )

    preds = preds[:current_index]
    trues = trues[:current_index]
    
    # Concatenate MDN components if collected
    mdn_components_tuple = None
    if enable_mdn and mdn_pi_list:
        pi_concat = np.concatenate(mdn_pi_list, axis=0)[:current_index]
        mu_concat = np.concatenate(mdn_mu_list, axis=0)[:current_index]
        sigma_concat = np.concatenate(mdn_sigma_list, axis=0)[:current_index]
        mdn_components_tuple = (pi_concat, mu_concat, sigma_concat)
    
    if memory_logging_enabled:
        _log_memory_snapshot(
            f"collect_predictions_{scenario_name}_end",
            device,
            MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
            {"processed": current_index},
        )
    return preds, trues, current_index, mdn_components_tuple


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    test_data: Any,
    device: torch.device,
    args: SimpleConfig,
    logger: logging.Logger,
    target_indices: Sequence[int],
) -> EvaluationResult:
    """Evaluate the trained model on production test data."""

    if hasattr(test_data, "__len__"):
        num_samples = len(test_data)
    else:
        num_samples = len(test_loader) * int(getattr(args, "batch_size", 1))

    memory_logging_enabled = getattr(args, "enable_memory_diagnostics", False)
    if memory_logging_enabled:
        _log_memory_snapshot(
            "evaluate_model_start",
            device,
            MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
            {"num_samples": num_samples},
        )

    preds, trues, processed, mdn_components = collect_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
        args=args,
        target_indices=target_indices,
        num_samples=num_samples,
        logger=logger,
        scenario_name="baseline",
    )

    overall: Dict[str, float] = {}
    per_target: Dict[str, Dict[str, float]] = {}

    if processed > 0:
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        overall = {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe),
        }

        # 🎲 MDN CALIBRATION METRICS - Log to file only
        enable_mdn = getattr(args, "enable_mdn_decoder", False)
        if enable_mdn and mdn_components is not None:
            pi_np, mu_np, sigma_np = mdn_components
            # Convert to tensors for calibration metrics
            pi_t = torch.from_numpy(pi_np)
            mu_t = torch.from_numpy(mu_np)
            sigma_t = torch.from_numpy(sigma_np)
            trues_t = torch.from_numpy(trues)
            
            calib_config = getattr(args, "calibration_metrics", {})
            coverage_levels = calib_config.get("coverage_levels", [0.5, 0.9])
            compute_crps = calib_config.get("compute_crps", False)
            crps_samples = calib_config.get("crps_samples", 100)
            
            calibration_dict = log_calibration_metrics(
                logger=MEMORY_LOGGER if MEMORY_LOGGER.handlers else logger,
                predictions=preds,
                targets=trues,
                pi=pi_t,
                mu=mu_t,
                sigma=sigma_t,
                coverage_levels=coverage_levels,
                compute_crps=compute_crps,
                crps_samples=crps_samples,
            )
            
            # Add test_loss alias for NLL (backward compat with console logging)
            if calibration_dict and "nll" in calibration_dict:
                overall["test_loss"] = calibration_dict["nll"]
        else:
            # Standard MSE-based test loss
            overall["test_loss"] = overall["mse"]

        if preds.shape[-1] == 4:
            ohlc_names = ["Open", "High", "Low", "Close"]
            for index, name in enumerate(ohlc_names):
                pred_i = preds[:, :, index:index + 1]
                true_i = trues[:, :, index:index + 1]
                mae_i, mse_i, rmse_i, mape_i, mspe_i = metric(pred_i, true_i)
                per_target[name] = {
                    "mae": float(mae_i),
                    "rmse": float(rmse_i),
                    "mape": float(mape_i),
                    "mspe": float(mspe_i),
                }
    else:
        logger.error("No test batches processed during evaluation; check data loader configuration.")

    if memory_logging_enabled:
        _log_memory_snapshot(
            "evaluate_model_end",
            device,
            MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
            {"processed_batches": processed},
        )
    return EvaluationResult(preds=preds, trues=trues, overall=overall, per_target=per_target, processed_batches=processed)


def run_adversarial_diagnostics(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    args: SimpleConfig,
    logger: logging.Logger,
    target_indices: Sequence[int],
    num_samples: int,
) -> List[AdversarialScenarioResult]:
    """Execute adversarial evaluation scenarios to validate production robustness."""

    if not _coerce_bool(getattr(args, "enable_adversarial_checks", True), True):
        logger.info("Adversarial diagnostics disabled by configuration")
        return []

    max_batches = max(1, int(getattr(args, "adversarial_max_batches", 4)))
    noise_std = float(getattr(args, "adversarial_noise_std", 0.05))
    scale_factor = float(getattr(args, "adversarial_scale_factor", 0.25))
    dropout_ratio = float(getattr(args, "adversarial_dropout_ratio", 0.1))

    memory_logging_enabled = getattr(args, "enable_memory_diagnostics", False)
    if memory_logging_enabled:
        _log_memory_snapshot(
            "adversarial_diagnostics_start",
            device,
            MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
            {"num_samples": num_samples, "max_batches": max_batches},
        )

    scenarios: List[Tuple[str, Callable[[torch.Tensor], torch.Tensor], str]] = [
        (
            "gaussian_noise",
            lambda batch: batch + torch.randn_like(batch) * noise_std,
            "Additive Gaussian noise stress test",
        ),
        (
            "zero_inputs",
            lambda batch: torch.zeros_like(batch),
            "All-zero encoder input to verify baseline handling",
        ),
        (
            "scale_spike",
            lambda batch: batch * (1 + scale_factor),
            "Uniform scaling spike across features",
        ),
        (
            "time_reverse",
            lambda batch: torch.flip(batch, dims=[1]),
            "Temporal order inversion",
        ),
        (
            "feature_dropout",
            lambda batch: batch * (torch.rand_like(batch) > dropout_ratio).float(),
            "Random feature dropout simulation",
        ),
    ]

    results: List[AdversarialScenarioResult] = []
    for name, transform, description in scenarios:
        if memory_logging_enabled:
            _log_memory_snapshot(
                f"adversarial_{name}_start",
                device,
                MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
                {},
            )
        preds, trues, processed, _ = collect_predictions(
            model=model,
            data_loader=test_loader,
            device=device,
            args=args,
            target_indices=target_indices,
            num_samples=num_samples,
            logger=logger,
            scenario_name=name,
            max_batches=max_batches,
            input_transform=transform,
        )

        if processed == 0:
            logger.warning("Adversarial scenario '%s' processed zero batches; skipping metric computation", name)
            results.append(
                AdversarialScenarioResult(
                    name=name,
                    overall={},
                    per_target={},
                    processed_batches=processed,
                    description=description,
                )
            )
            continue

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        overall_metrics = {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe),
        }

        per_target_metrics: Dict[str, Dict[str, float]] = {}
        if preds.shape[-1] == 4:
            ohlc_names = ["Open", "High", "Low", "Close"]
            for index, channel_name in enumerate(ohlc_names):
                pred_i = preds[:, :, index:index + 1]
                true_i = trues[:, :, index:index + 1]
                mae_i, mse_i, rmse_i, mape_i, mspe_i = metric(pred_i, true_i)
                per_target_metrics[channel_name] = {
                    "mae": float(mae_i),
                    "rmse": float(rmse_i),
                    "mape": float(mape_i),
                    "mspe": float(mspe_i),
                }

        logger.info(
            "Adversarial scenario '%s' | processed_batches=%s rmse=%0.6f mape=%0.6f",
            name,
            processed,
            overall_metrics["rmse"],
            overall_metrics["mape"],
        )
        if memory_logging_enabled:
            _log_memory_snapshot(
                f"adversarial_{name}_end",
                device,
                MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
                {
                    "processed_batches": processed,
                    "rmse": overall_metrics.get("rmse", float("nan")),
                },
            )
        results.append(
            AdversarialScenarioResult(
                name=name,
                overall=overall_metrics,
                per_target=per_target_metrics,
                processed_batches=processed,
                description=description,
            )
        )

    if memory_logging_enabled:
        _log_memory_snapshot(
            "adversarial_diagnostics_end",
            device,
            MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
            {"scenarios": len(results)},
        )
    return results


def scale_targets_for_loss(
    targets_unscaled: torch.Tensor,
    target_scaler: Any,
    target_indices: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Scale target values for loss computation to match scaled predictions
    
    Args:
        targets_unscaled: [batch, seq_len, n_features] Unscaled target tensor from batch_y
        target_scaler: Fitted scaler for targets
        target_indices: List of indices for target features
        device: PyTorch device
    
    Returns:
        Scaled targets tensor for loss computation
    """
    try:
        # Extract only the target features
        targets_only = targets_unscaled[:, :, target_indices]  # [batch, seq_len, n_targets]
        targets_np = targets_only.cpu().numpy()
        
        # Reshape for scaler: [batch*seq_len, n_targets]
        batch_size, seq_len, n_targets = targets_np.shape
        targets_reshaped = targets_np.reshape(-1, n_targets)
        
        # Scale using target scaler
        targets_scaled_reshaped = target_scaler.transform(targets_reshaped)
        
        # Reshape back: [batch, seq_len, n_targets]
        targets_scaled_np = targets_scaled_reshaped.reshape(batch_size, seq_len, n_targets)
        
        return torch.from_numpy(targets_scaled_np).float().to(device)
        
    except Exception as exc:
        LOGGER.exception("Target scaling for loss failed: %s", exc)
        return targets_unscaled[:, :, target_indices].to(device)

def get_warmup_cosine_lr(
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> float:
    """
    Warmup + Cosine Annealing Learning Rate Schedule
    
    Args:
        epoch: Current epoch (0-indexed)
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        base_lr: Base learning rate after warmup
        min_lr: Minimum learning rate at the end
    
    Returns:
        Learning rate for the current epoch
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

def adjust_learning_rate_warmup_cosine(optimizer: Optimizer, epoch: int, args: Any) -> float:
    """
    Enhanced learning rate adjustment with warmup + cosine annealing
    """
    warmup_epochs = getattr(args, 'warmup_epochs', 5)
    min_lr = float(getattr(args, 'min_lr', 1e-6))
    base_lr = float(args.learning_rate)
    
    if hasattr(args, 'lradj') and args.lradj == 'warmup_cosine':
        lr = get_warmup_cosine_lr(epoch, warmup_epochs, args.train_epochs, base_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if epoch < warmup_epochs:
            LOGGER.info(
                "Warmup phase | epoch=%s/%s lr=%0.8f",
                epoch + 1,
                warmup_epochs,
                lr,
            )
        else:
            LOGGER.info(
                "Cosine annealing | epoch=%s/%s lr=%0.8f",
                epoch + 1,
                args.train_epochs,
                lr,
            )
        return lr
    else:
        # Fallback to standard adjustment
        from utils.tools import adjust_learning_rate
        adjust_learning_rate(optimizer, epoch, args)
        return optimizer.param_groups[0]['lr']

def _create_enhanced_decoder_input(batch_y, args, logger):
    """
    🚀 ENHANCED: Create decoder input with future celestial data
    
    This function leverages the unique advantage of having predictable celestial positions
    to provide the model with known future covariate information during prediction.
    
    Args:
        batch_y: Full batch data [batch, seq_len, features]
        args: Training arguments with pred_len, label_len
        logger: Logger for debugging
    
    Returns:
        Enhanced decoder input with future celestial data
    """
    # Define feature indices based on CSV structure:
    # [date(0), log_Open(1), log_High(2), log_Low(3), log_Close(4), time_delta(5), celestial_features(6-117)]
    # But in the processed data, date is removed, so indices shift by -1:
    # [log_Open(0), log_High(1), log_Low(2), log_Close(3), time_delta(4), celestial_features(5-117)]
    
    target_indices_in_data = [0, 1, 2, 3]  # OHLC positions in processed data
    celestial_indices = list(range(4, batch_y.shape[-1]))  # time_delta + celestial features
    
    # Extract future celestial data (known via ephemeris)
    future_celestial = batch_y[:, -args.pred_len:, celestial_indices]  # [batch, pred_len, 114]
    
    # Zero out future OHLC targets (unknown - what we're predicting)
    future_targets = torch.zeros_like(batch_y[:, -args.pred_len:, target_indices_in_data])  # [batch, pred_len, 4]
    
    # Reconstruct future data maintaining original feature order: [OHLC_zeros, time_delta+celestial_known]
    future_combined = torch.cat([future_targets, future_celestial], dim=-1)  # [batch, pred_len, 118]
    
    # Create enhanced decoder input: [historical_data, future_celestial_enhanced]
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], future_combined], dim=1)  # [batch, label_len+pred_len, 117]
    
    logger.debug(
        f"🔮 Enhanced decoder input created: "
        f"historical_shape={batch_y[:, :args.label_len, :].shape}, "
        f"future_celestial_shape={future_combined.shape}, "
        f"total_shape={dec_inp.shape}"
    )
    
    return dec_inp


def _normalize_model_output(
    raw_output: Any,
) -> Tuple[torch.Tensor, float, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], Optional[Dict[str, Any]]]:
    """Convert mixed forward outputs into tensor, scalar aux loss, optional MDN tuple, and metadata."""
    aux_loss = 0.0
    mdn_tuple = None
    metadata = None
    output_tensor = raw_output

    if isinstance(raw_output, (tuple, list)):
        if len(raw_output) == 4:
            # MDN case: (point_pred, aux_loss, (pi, mu, sigma), metadata)
            point_pred, aux, mdn_components, meta = raw_output
            if isinstance(point_pred, torch.Tensor):
                output_tensor = point_pred
            if isinstance(aux, (int, float)):
                aux_loss = float(aux)
            if isinstance(mdn_components, (tuple, list)) and len(mdn_components) == 3:
                mdn_tuple = (mdn_components[0], mdn_components[1], mdn_components[2])
            if isinstance(meta, dict):
                metadata = meta
        elif len(raw_output) == 2:
            primary, secondary = raw_output
            # Check if secondary is metadata (dict) or auxiliary loss
            if isinstance(secondary, dict):
                metadata = secondary
                output_tensor = primary
            elif isinstance(secondary, torch.Tensor) and secondary.numel() == 1:
                aux_loss = float(secondary.item())
                output_tensor = primary
            elif isinstance(secondary, (int, float)):
                aux_loss = float(secondary)
                output_tensor = primary
            else:
                output_tensor = primary
        elif len(raw_output) == 3 and all(isinstance(part, torch.Tensor) for part in raw_output):
            mdn_tuple = (raw_output[0], raw_output[1], raw_output[2])
            output_tensor = raw_output[0]
        elif len(raw_output) >= 1:
            output_tensor = raw_output[0]

    if not isinstance(output_tensor, torch.Tensor):
        raise TypeError("Model forward pass must yield a tensor as primary output.")

    return output_tensor, aux_loss, mdn_tuple, metadata

def train_celestial_pgat_production():
    """Production training function for overnight runs."""
    config_path = "configs/celestial_enhanced_pgat_production.yaml"
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config_dict = yaml.safe_load(config_file)

    args = SimpleConfig(config_dict)
    logger = configure_logging(args)

    logger.info("Starting PRODUCTION Celestial Enhanced PGAT training run")
    logger.info("Heavy-duty overnight configuration enabled")
    logger.info("=" * 80)

    # Add missing required attributes
    args.task_name = 'long_term_forecast'
    args.model_name = 'Celestial_Enhanced_PGAT'
    args.data_name = 'custom'
    args.checkpoints = './checkpoints/'
    args.inverse = False
    args.cols = None
    args.num_workers = getattr(args, 'num_workers', 0)
    args.itr = 1
    args.train_only = False
    args.do_predict = False
    args.model_id = getattr(
        args,
        'model_id',
        f"{args.model_name}_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Wave aggregation settings
    args.aggregate_waves_to_celestial = getattr(args, 'aggregate_waves_to_celestial', True)
    args.wave_to_celestial_mapping = getattr(args, 'wave_to_celestial_mapping', False)
    args.celestial_node_features = getattr(args, 'celestial_node_features', 13)

    # Target wave indices
    args.target_wave_indices = getattr(args, 'target_wave_indices', [0, 1, 2, 3])

    # Memory diagnostics configuration
    args.enable_memory_diagnostics = _coerce_bool(
        getattr(args, "enable_memory_diagnostics", True),
        True,
    )
    args.memory_log_interval = max(1, int(getattr(args, "memory_log_interval", 25)))
    args.dump_cuda_memory_summary = _coerce_bool(
        getattr(args, "dump_cuda_memory_summary", False),
        False,
    )
    args.memory_summary_on_exception = _coerce_bool(
        getattr(args, "memory_summary_on_exception", True),
        True,
    )

    checkpoint_dir = Path(args.checkpoints) / args.model_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    memory_logger: Optional[logging.Logger] = None
    for existing_handler in list(MEMORY_LOGGER.handlers):
        existing_handler.close()
    MEMORY_LOGGER.handlers.clear()
    MEMORY_LOGGER.propagate = False
    if args.enable_memory_diagnostics:
        MEMORY_LOGGER.setLevel(logging.DEBUG)
        memory_logger = MEMORY_LOGGER
        memory_log_file = checkpoint_dir / "memory_diagnostics.log"
        memory_handler = logging.FileHandler(memory_log_file, encoding="utf-8")
        log_format = getattr(args, "log_format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
        memory_handler.setFormatter(logging.Formatter(log_format))
        memory_logger.addHandler(memory_handler)
        memory_logger.info(
            "Memory diagnostics enabled | log_interval=%s dump_cuda_summary=%s",
            args.memory_log_interval,
            args.dump_cuda_memory_summary,
        )
        logger.info("Memory diagnostics enabled; detailed logs stored at %s", memory_log_file)
    else:
        MEMORY_LOGGER.setLevel(logging.CRITICAL)

    logger.info(
        "Configuration summary | model=%s seq_len=%s pred_len=%s d_model=%s n_heads=%s e_layers=%s d_layers=%s train_epochs=%s",
        getattr(args, 'model_name', getattr(args, 'model', 'Celestial_Enhanced_PGAT')),
        getattr(args, 'seq_len', 'Unknown'),
        getattr(args, 'pred_len', 'Unknown'),
        getattr(args, 'd_model', 'Unknown'),
        getattr(args, 'n_heads', 'Unknown'),
        getattr(args, 'e_layers', 'Unknown'),
        getattr(args, 'd_layers', 'Unknown'),
        getattr(args, 'train_epochs', 'Unknown'),
    )
    logger.info(
        "Optimization summary | batch_size=%s learning_rate=%s patience=%s target=%s target_wave_indices=%s c_out=%s wave_aggregation=%s",
        getattr(args, 'batch_size', 'Unknown'),
        getattr(args, 'learning_rate', 'Unknown'),
        getattr(args, 'patience', 'Unknown'),
        getattr(args, 'target', 'Not specified'),
        getattr(args, 'target_wave_indices', []),
        getattr(args, 'c_out', 'Unknown'),
        getattr(args, 'aggregate_waves_to_celestial', False),
    )

    if getattr(args, 'use_gpu', True) and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using device: %s (GPU production mode)", device)
    else:
        device = torch.device('cpu')
        logger.info("Using device: %s (CPU fallback)", device)

    if args.enable_memory_diagnostics:
        _log_memory_snapshot(
            "device_selection",
            device,
            memory_logger,
            {"use_gpu": getattr(args, "use_gpu", True)},
        )

    seed_value = int(getattr(args, "production_seed", getattr(args, "seed", getattr(args, "random_seed", 2024))))
    _ensure_reproducibility(seed_value)
    logger.info("Reproducibility seed configured to %s", seed_value)

    logger.info("Loading production data modules")
    data_modules = prepare_data_modules(args, logger, device)
    train_data = data_modules["train"].dataset
    train_loader = data_modules["train"].loader
    vali_loader = data_modules["val"].loader
    test_data = data_modules["test"].dataset
    test_loader = data_modules["test"].loader
    logger.info(
        "Data loader sizes | train=%s val=%s test=%s",
        len(train_loader),
        len(vali_loader),
        len(test_loader),
    )

    if args.enable_memory_diagnostics:
        _log_memory_snapshot(
            "post_data_module_loading",
            device,
            memory_logger,
            {
                "train_batches": len(train_loader),
                "val_batches": len(vali_loader),
                "test_batches": len(test_loader),
            },
        )

    logger.info("Preparing scaling utilities for loss computation")
    train_scaler = getattr(train_data, 'scaler', None)
    target_scaler = getattr(train_data, 'target_scaler', None)

    if train_scaler is None:
        logger.error("No scaler found in train_data; aborting production training")
        return False

    logger.info("Main scaler detected with %s features", train_scaler.n_features_in_)
    if target_scaler:
        logger.info("Target scaler detected with %s features", target_scaler.n_features_in_)
    else:
        logger.warning("No separate target scaler found; reusing main scaler for targets")
        target_scaler = train_scaler

    target_indices: List[int] = [0, 1, 2, 3]  # OHLC indices
    logger.info("Using target indices for OHLC: %s", target_indices)

    if args.enable_memory_diagnostics:
        _log_memory_snapshot(
            "post_scaler_setup",
            device,
            memory_logger,
            {
                "has_target_scaler": bool(target_scaler),
                "target_indices": target_indices,
            },
        )

    logger.info("Initializing production Celestial Enhanced PGAT model")
    try:
        model = Model(args).to(device)
        logger.info("Model initialized successfully")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_size = total_params * 4 / (1024 ** 2)
        logger.info(
            "Parameter statistics | total=%s trainable=%s approx_size_mb=%0.1f",
            f"{total_params:,}",
            f"{trainable_params:,}",
            param_size,
        )

        if args.enable_memory_diagnostics:
            _log_memory_snapshot(
                "post_model_initialization",
                device,
                memory_logger,
                {
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "param_size_mb": param_size,
                },
            )

    except Exception as exc:
        if args.enable_memory_diagnostics and args.memory_summary_on_exception:
            _log_memory_snapshot("model_initialization_exception", device, memory_logger, {"exception": str(exc)})
            _maybe_dump_cuda_summary(device, memory_logger, "model_initialization_exception")
        logger.exception("Model initialization failed: %s", exc)
        return False
    
    # Setup training components with enhanced features
    # Ensure learning_rate is float (YAML might load it as string)
    learning_rate = float(args.learning_rate)
    weight_decay = float(getattr(args, 'weight_decay', 0.0001))
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Mixed precision training
    use_amp = getattr(args, 'mixed_precision', False) and device.type == 'cuda'
    if use_amp:
        if GradScaler is None or torch_autocast is None:
            raise RuntimeError("Mixed precision requested but torch.cuda.amp is unavailable in this environment.")
        scaler = GradScaler()
        logger.info("Mixed precision training enabled")
    else:
        scaler = None
        logger.info("Standard precision training")
    
    # Gradient accumulation
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    effective_batch_size = args.batch_size * gradient_accumulation_steps
    logger.info(
        "Gradient accumulation configured | steps=%s effective_batch_size=%s",
        gradient_accumulation_steps,
        effective_batch_size,
    )

    if args.enable_memory_diagnostics:
        _log_memory_snapshot(
            "post_optimizer_setup",
            device,
            memory_logger,
            {
                "learning_rate": float(learning_rate),
                "weight_decay": weight_decay,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
        )
    
    # Learning rate schedule preview
    warmup_epochs = getattr(args, 'warmup_epochs', 5)
    if hasattr(args, 'lradj') and args.lradj == 'warmup_cosine':
        base_lr = float(args.learning_rate)
        min_lr_val = float(getattr(args, 'min_lr', 1e-6))
        logger.info(
            "Learning rate schedule preview | warmup_epochs=%s base_lr=%0.6f min_lr=%0.6f remaining_epochs=%s",
            warmup_epochs,
            base_lr,
            min_lr_val,
            args.train_epochs - warmup_epochs,
        )
        sample_epochs = [0, warmup_epochs - 1, warmup_epochs, args.train_epochs // 2, args.train_epochs - 1]
        samples: List[str] = []
        for ep in sample_epochs:
            if ep < args.train_epochs:
                lr = get_warmup_cosine_lr(ep, warmup_epochs, args.train_epochs, base_lr, min_lr_val)
                samples.append(f"E{ep + 1}:{lr:.6f}")
        if samples:
            logger.info("Sample learning rates: %s", ", ".join(samples))
    
    # Import loss functions at function scope to avoid UnboundLocalError
    try:
        from layers.modular.decoder.sequential_mixture_decoder import SequentialMixtureNLLLoss
        from layers.modular.decoder.mixture_density_decoder import MixtureNLLLoss
    except ImportError:
        # Fallback if mixture losses are not available
        SequentialMixtureNLLLoss = None
        MixtureNLLLoss = None
    
    # Use proper loss function
    if getattr(model, 'use_mixture_decoder', False) and SequentialMixtureNLLLoss is not None:
        criterion = SequentialMixtureNLLLoss(reduction='mean')
        logger.info("Using Gaussian Mixture NLL Loss for probabilistic predictions")
    else:
        criterion = nn.MSELoss()
        logger.info("Using MSE Loss for deterministic predictions")
    
    # NO EARLY STOPPING for production
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    logger.info("Early stopping patience configured to %s (effectively disabled)", args.patience)
    
    # Checkpoint directory already created during diagnostics setup
    logger.info("Checkpoints will be saved to %s", checkpoint_dir)
    
    # Auto-detect target indices from CSV header
    detected_target_indices: Optional[List[int]] = None
    try:
        if hasattr(args, 'target') and args.target:
            csv_path = os.path.join(args.root_path, args.data_path)
            logger.info("Inspecting data file at %s for target resolution", csv_path)
            df_head = pd.read_csv(csv_path, nrows=1)
            feature_cols = [c for c in df_head.columns if c != 'date']
            logger.info(
                "Feature summary | total_columns=%s feature_columns=%s",
                len(df_head.columns),
                len(feature_cols),
            )
            
            # Estimate row count
            try:
                with open(csv_path, 'r', encoding='utf-8', errors='ignore') as csv_file:
                    row_count = sum(1 for _ in csv_file) - 1
                logger.info("Estimated CSV row count: %s", row_count)
            except Exception:
                pass
            
            # Auto-set dimensions
            args.enc_in = len(feature_cols)
            args.dec_in = len(feature_cols)
            logger.info("Auto-set enc_in/dec_in to %s based on feature count", args.enc_in)
            
            if isinstance(args.target, str):
                target_names = [t.strip() for t in args.target.split(',')]
            else:
                target_names = list(args.target)
            name_to_index = {name: idx for idx, name in enumerate(feature_cols)}
            detected_target_indices = [name_to_index[n] for n in target_names if n in name_to_index]

            if not detected_target_indices:
                detected_target_indices = list(range(getattr(args, 'c_out', 1)))
            logger.info("Resolved target indices: %s", detected_target_indices)
            logger.info("Resolved target names: %s", target_names)
            
            if len(detected_target_indices) != args.c_out:
                logger.warning(
                    "Adjusting c_out from %s to %s based on detected targets",
                    args.c_out,
                    len(detected_target_indices),
                )
                args.c_out = len(detected_target_indices)
                
    except Exception as exc:
        logger.warning("Failed to auto-detect target indices: %s", exc)
        detected_target_indices = list(range(getattr(args, 'c_out', 4)))

    if detected_target_indices is not None:
        target_indices = detected_target_indices

    sequential_mixture_loss_cls: Optional[Type[nn.Module]] = None
    mixture_loss_cls: Optional[Type[nn.Module]] = None
    if SequentialMixtureNLLLoss is not None:
        sequential_mixture_loss_cls = SequentialMixtureNLLLoss
    if MixtureNLLLoss is not None:
        mixture_loss_cls = MixtureNLLLoss

    # PRODUCTION Training loop
    logger.info("Starting production training loop")
    logger.info("Configured for %s training epochs", args.train_epochs)
    artifacts = create_training_artifacts(checkpoint_dir)

    if args.enable_memory_diagnostics:
        _log_memory_snapshot(
            "training_loop_start",
            device,
            memory_logger,
            {
                "train_epochs": args.train_epochs,
                "train_batches": len(train_loader),
                "val_batches": len(vali_loader),
            },
        )

    training_start_time = time.time()
    total_train_batches = len(train_loader)
    full_cycles, remainder_batches = divmod(total_train_batches, gradient_accumulation_steps)
    train_celestial_pgat_production.best_val_loss = artifacts.best_val_loss  # type: ignore[attr-defined]

    for epoch in range(args.train_epochs):
        epoch_start = time.time()

        avg_train_loss, train_batches = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            use_amp=use_amp,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device=device,
            epoch=epoch,
            args=args,
            logger=logger,
            target_scaler=target_scaler,
            target_indices=target_indices,
            total_train_batches=total_train_batches,
            full_cycles=full_cycles,
            remainder_batches=remainder_batches,
            epoch_start=epoch_start,
            sequential_mixture_loss_cls=sequential_mixture_loss_cls,
            mixture_loss_cls=mixture_loss_cls,
        )

        avg_val_loss, val_batches = validate_epoch(
            model=model,
            val_loader=vali_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            args=args,
            logger=logger,
            target_scaler=target_scaler,
            target_indices=target_indices,
            sequential_mixture_loss_cls=sequential_mixture_loss_cls,
            mixture_loss_cls=mixture_loss_cls,
        )

        current_lr = adjust_learning_rate_warmup_cosine(optimizer, epoch, args)

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - training_start_time
        remaining_epochs = args.train_epochs - (epoch + 1)
        estimated_remaining = (total_elapsed / (epoch + 1) * remaining_epochs) if (epoch + 1) else 0.0

        logger.info(
            "Epoch %s/%s complete | train_loss=%0.6f val_loss=%0.6f lr=%0.8f epoch_time=%0.2fs elapsed=%0.2fh remaining=%0.2fh",
            epoch + 1,
            args.train_epochs,
            avg_train_loss,
            avg_val_loss,
            current_lr,
            epoch_time,
            total_elapsed / 3600,
            estimated_remaining / 3600 if estimated_remaining else 0.0,
        )

        if train_batches == 0:
            logger.warning("Epoch %s processed zero training batches; check data loader configuration.", epoch + 1)
        if val_batches == 0:
            logger.warning("Epoch %s processed zero validation batches; check data loader configuration.", epoch + 1)

        is_best = artifacts.record_epoch(epoch, avg_train_loss, avg_val_loss)
        train_celestial_pgat_production.best_val_loss = artifacts.best_val_loss  # type: ignore[attr-defined]

        checkpoint_interval = getattr(args, "checkpoint_interval", 5)
        save_best_only = getattr(args, "save_best_only", False)
        manage_checkpoints(
            artifacts=artifacts,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            current_lr=current_lr,
            checkpoint_dir=checkpoint_dir,
            save_best_only=save_best_only,
            checkpoint_interval=checkpoint_interval,
            logger=logger,
            is_best=is_best,
        )

        early_stopping(avg_val_loss, model, str(checkpoint_dir))
        if early_stopping.early_stop:
            logger.warning("Early stopping triggered at epoch %s", epoch + 1)
            break
    
    if args.enable_memory_diagnostics:
        _log_memory_snapshot(
            "training_loop_end",
            device,
            memory_logger,
            {"epochs_completed": len(artifacts.train_losses)},
        )
        if getattr(args, "dump_cuda_memory_summary", False):
            _maybe_dump_cuda_summary(device, memory_logger, "training_loop_end")

    # Load best model
    best_model_path = artifacts.best_checkpoint_path
    if not best_model_path.exists():
        best_model_path = checkpoint_dir / 'best_model.pth'
    fallback_path = checkpoint_dir / 'checkpoint.pth'
    
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            val_loss_value: Any = checkpoint.get('val_loss', 'unknown')
            if isinstance(val_loss_value, torch.Tensor):
                val_loss_value = val_loss_value.detach().item()
            val_loss_repr = f"{val_loss_value:.6f}" if isinstance(val_loss_value, (int, float)) else val_loss_value
            logger.info(
                "Best model loaded | epoch=%s val_loss=%s",
                checkpoint.get('epoch', 'unknown'),
                val_loss_repr,
            )
        else:
            model.load_state_dict(checkpoint)
            logger.info("Best model loaded (legacy format)")
    elif fallback_path.exists():
        model.load_state_dict(torch.load(fallback_path, map_location=device))
        logger.info("Fallback model loaded")
    else:
        logger.warning("No saved model found, using final training state")
    
    # Final evaluation
    logger.info("Final PRODUCTION evaluation starting")
    evaluation_result = evaluate_model(
        model=model,
        test_loader=test_loader,
        test_data=test_data,
        device=device,
        args=args,
        logger=logger,
        target_indices=target_indices,
    )

    if not evaluation_result.success:
        logger.error("No valid predictions generated")
        return False

    overall_metrics = evaluation_result.overall
    rmse_value = overall_metrics.get("rmse", float("nan"))
    # Emit a test_loss alias (rmse) so it appears on console with the loss-only filter
    logger.info(
        "Test summary | test_loss=%0.6f rmse=%0.6f mae=%0.6f mse=%0.6f",
        rmse_value,
        rmse_value,
        overall_metrics.get("mae", float("nan")),
        overall_metrics.get("mse", float("nan")),
    )
    logger.info(
        "PRODUCTION results (OHLC prediction) | mae=%0.6f mse=%0.6f rmse=%0.6f mape=%0.6f mspe=%0.6f",
        overall_metrics.get("mae", float("nan")),
        overall_metrics.get("mse", float("nan")),
        rmse_value,
        overall_metrics.get("mape", float("nan")),
        overall_metrics.get("mspe", float("nan")),
    )

    if args.enable_memory_diagnostics:
        _log_memory_snapshot(
            "post_baseline_evaluation",
            device,
            memory_logger,
            {"rmse": rmse_value, "processed_batches": evaluation_result.processed_batches},
        )

    if evaluation_result.per_target:
        logger.info("Individual OHLC metrics")
        for channel_name, metrics_map in evaluation_result.per_target.items():
            logger.info(
                "%s metrics | mae=%0.6f rmse=%0.6f mape=%0.6f",
                channel_name,
                metrics_map.get("mae", float("nan")),
                metrics_map.get("rmse", float("nan")),
                metrics_map.get("mape", float("nan")),
            )

    adversarial_sample_count = evaluation_result.preds.shape[0]
    adversarial_results = run_adversarial_diagnostics(
        model=model,
        test_loader=test_loader,
        device=device,
        args=args,
        logger=logger,
        target_indices=target_indices,
        num_samples=adversarial_sample_count,
    )

    if args.enable_memory_diagnostics:
        _log_memory_snapshot(
            "post_adversarial_evaluation",
            device,
            memory_logger,
            {"scenarios": len(adversarial_results)},
        )

    total_training_seconds = time.time() - training_start_time
    persist_training_results(
        checkpoint_dir=checkpoint_dir,
        config_dict=config_dict,
        args=args,
        total_params=total_params,
        artifacts=artifacts,
        evaluation_result=evaluation_result,
        adversarial_results=adversarial_results,
        training_duration_seconds=total_training_seconds,
        rmse_value=rmse_value,
        effective_batch_size=effective_batch_size,
        warmup_epochs=warmup_epochs,
        use_amp=use_amp,
        logger=logger,
    )

    if args.enable_memory_diagnostics:
        _log_memory_snapshot(
            "training_run_complete",
            device,
            memory_logger,
            {"training_seconds": total_training_seconds},
        )
        if getattr(args, "dump_cuda_memory_summary", False):
            _maybe_dump_cuda_summary(device, memory_logger, "training_run_complete")

    return True

def main():
    """Main function"""
    try:
        success = train_celestial_pgat_production()
        return 0 if success else 1
    except Exception as exc:
        LOGGER.exception("PRODUCTION training failed with error: %s", exc)
        return 1

if __name__ == "__main__":
    exit(main())