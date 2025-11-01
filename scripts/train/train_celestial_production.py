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

# AMP dtype resolver
def _resolve_amp_dtype(args: "SimpleConfig") -> "torch.dtype":
    """Pick an AMP dtype based on config and hardware.

    Honors args.amp_dtype if provided ("float16"/"fp16" or "bfloat16"/"bf16").
    Otherwise prefers bfloat16 when supported, else float16.
    """
    import torch
    requested = str(getattr(args, "amp_dtype", "auto")).lower()
    if requested in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if requested in {"fp16", "float16"}:
        return torch.float16
    # auto mode
    try:
        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16
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
# Import the new modular model instead of the monolithic one
from models.Celestial_Enhanced_PGAT_Modular import Model
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
        # Prevent infinite recursion during pickling
        if name == '_data':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
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
    
    def __getstate__(self):
        """Support for pickling (needed for multiprocessing)"""
        return {'_data': self._data}
    
    def __setstate__(self, state):
        """Support for unpickling (needed for multiprocessing)"""
        super().__setattr__("_data", state['_data'])


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
            # Allow: per-batch loss, epoch summaries, directional accuracy, validation, and test loss
            if "loss=" in msg:
                return True
            if msg.startswith("Epoch ") and ("complete" in msg or "production training" in msg):
                return True
            if "test_loss=" in msg:
                return True
            if "dir_acc=" in msg:
                return True
            if "DIRECTIONAL ACCURACY" in msg:
                return True
            if "LOSSES:" in msg:
                return True
            if "val_loss=" in msg:
                return True
            if "Starting validation" in msg:
                return True
            if "Validation complete" in msg:
                return True
            if "TREND:" in msg:
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
    directional_accuracies: List[Optional[float]] = None  # type: ignore[assignment]
    best_directional_accuracy: Optional[float] = None

    def __post_init__(self):
        """Initialize directional_accuracies list if not provided."""
        if self.directional_accuracies is None:
            self.directional_accuracies = []

    def record_epoch(self, epoch: int, train_loss: float, val_loss: float, dir_accuracy: Optional[float] = None) -> bool:
        """Store epoch metrics and return True when validation loss improves."""

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.directional_accuracies.append(dir_accuracy)
        
        if dir_accuracy is not None:
            if self.best_directional_accuracy is None or dir_accuracy > self.best_directional_accuracy:
                self.best_directional_accuracy = dir_accuracy
        
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


def _compute_detailed_batch_metrics(
    raw_loss: torch.Tensor,
    outputs_tensor: torch.Tensor,
    y_true_for_loss: torch.Tensor,
    mdn_outputs: Optional[Tuple],
    args
) -> dict:
    """Compute detailed batch metrics including loss components and directional accuracy."""
    metrics = {
        'total_loss': float(raw_loss.detach().item()),
        'mse_component': 0.0,
        'directional_component': 0.0,
        'mdn_component': 0.0,
        'directional_accuracy': None,
        'trend_accuracy': None,
        'sign_accuracy': None,
    }
    
    try:
        # Import loss functions for component analysis
        from layers.modular.losses.directional_trend_loss import (
            compute_directional_accuracy,
            DirectionalTrendLoss,
            HybridMDNDirectionalLoss
        )
        
        # Compute directional accuracy
        if outputs_tensor is not None and y_true_for_loss is not None:
            # Use MDN mean if available, otherwise use direct outputs
            pred_for_accuracy = outputs_tensor
            if mdn_outputs is not None:
                try:
                    # Extract mean from MDN outputs
                    pi, mu, sigma = mdn_outputs
                    pred_for_accuracy = (pi * mu).sum(dim=-1)  # Weighted mean
                except Exception:
                    pass  # Use direct outputs as fallback
            
            # Compute directional accuracy
            dir_acc = compute_directional_accuracy(pred_for_accuracy, y_true_for_loss)
            metrics['directional_accuracy'] = float(dir_acc)
            
            # Compute additional accuracy metrics
            metrics.update(_compute_additional_accuracy_metrics(pred_for_accuracy, y_true_for_loss))
        
        # Decompose loss components if using hybrid loss
        loss_config = getattr(args, 'loss', {})
        if isinstance(loss_config, dict) and loss_config.get('type') == 'hybrid_mdn_directional':
            metrics.update(_decompose_hybrid_loss_components(
                outputs_tensor, y_true_for_loss, mdn_outputs, loss_config
            ))
    
    except Exception as e:
        # Graceful fallback if detailed metrics computation fails
        print(f"Warning: Detailed metrics computation failed: {e}")
        pass
    
    return metrics


def _compute_additional_accuracy_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute additional accuracy metrics like trend and sign accuracy."""
    metrics = {}
    
    try:
        # Sign accuracy (percentage of correct signs)
        pred_signs = torch.sign(pred)
        target_signs = torch.sign(target)
        sign_matches = (pred_signs == target_signs).float()
        metrics['sign_accuracy'] = float(sign_matches.mean() * 100)
        
        # Trend accuracy (correlation between predicted and actual changes)
        if pred.shape[1] > 1:  # Need at least 2 timesteps for trend
            pred_diff = pred[:, 1:] - pred[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            
            # Compute correlation
            pred_flat = pred_diff.flatten()
            target_flat = target_diff.flatten()
            
            if len(pred_flat) > 1:
                correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
                if not torch.isnan(correlation):
                    metrics['trend_accuracy'] = float(correlation * 100)
    
    except Exception:
        pass
    
    return metrics


def _decompose_hybrid_loss_components(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mdn_outputs: Optional[Tuple],
    loss_config: dict
) -> dict:
    """Decompose hybrid loss into its components."""
    components = {}
    
    try:
        # MSE component
        mse_loss = torch.nn.functional.mse_loss(outputs, targets)
        components['mse_component'] = float(mse_loss.item()) * loss_config.get('mse_weight', 0.6)
        
        # Directional component (approximate)
        if outputs.shape[1] > 1:
            pred_diff = outputs[:, 1:] - outputs[:, :-1]
            target_diff = targets[:, 1:] - targets[:, :-1]
            
            # Simple directional loss approximation
            pred_signs = torch.sign(pred_diff)
            target_signs = torch.sign(target_diff)
            directional_loss = torch.nn.functional.mse_loss(pred_signs, target_signs)
            components['directional_component'] = float(directional_loss.item()) * loss_config.get('directional_weight', 0.25)
        
        # MDN component (if available)
        if mdn_outputs is not None:
            try:
                from layers.modular.decoder.mdn_decoder import mdn_nll_loss
                pi, mu, sigma = mdn_outputs
                mdn_loss = mdn_nll_loss(pi, mu, sigma, targets)
                components['mdn_component'] = float(mdn_loss.item()) * loss_config.get('mdn_weight', 0.15)
            except Exception:
                pass
    
    except Exception:
        pass
    
    return components


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

    # Fallback: if the rebuilt loader yields zero batches, relax constraints
    try:
        if len(optimized_loader) == 0:
            logger.warning(
                "Optimized %s loader produced zero batches (batch_size=%s, drop_last=%s). "
                "Rebuilding with drop_last=False and a safer batch_size.",
                flag,
                loader_kwargs.get("batch_size"),
                loader_kwargs.get("drop_last"),
            )
            safe_bs = max(1, min(int(loader_kwargs.get("batch_size", 1)), len(dataset))) if hasattr(dataset, "__len__") else 1
            loader_kwargs["batch_size"] = safe_bs
            loader_kwargs["drop_last"] = False
            optimized_loader = DataLoader(dataset, **loader_kwargs)
    except Exception:
        # If len() is unsupported by custom loader, skip fallback
        pass

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


def data_provider_with_scaler(args: SimpleConfig, flag: str, scaler=None, target_scaler=None):
    """
    FIXED: Enhanced data_provider that passes fitted scalers to val/test datasets
    
    This ensures validation and test datasets use the same scaling as training data,
    preventing the critical bug where models trained on scaled data receive unscaled
    validation/test inputs.
    """
    from data_provider.data_factory import data_dict
    
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # Create dataset with fitted scalers passed from training
    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=getattr(args, 'seasonal_patterns', 'Monthly'),
        scaler=scaler,  # FIXED: Pass fitted scaler from training
        target_scaler=target_scaler  # FIXED: Pass fitted target_scaler from training
    )
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    
    return data_set, data_loader


def prepare_data_modules(
    args: SimpleConfig,
    logger: logging.Logger,
    device: torch.device,
) -> Dict[str, DataModule]:
    """Load datasets and data loaders with validation and diagnostic logging."""

    args.num_workers = _determine_num_workers(args, logger)

    modules: Dict[str, DataModule] = {}
    train_scaler = None
    train_target_scaler = None
    
    # FIXED: Create datasets in order and pass scalers from training to val/test
    for flag in ("train", "val", "test"):
        if flag == "train":
            # Create training dataset first to get the fitted scalers
            dataset, loader = data_provider(args, flag=flag)
            train_scaler = getattr(dataset, 'scaler', None)
            train_target_scaler = getattr(dataset, 'target_scaler', None)
            logger.info(f"Training scalers extracted: main_scaler={train_scaler is not None}, target_scaler={train_target_scaler is not None}")
        else:
            # For val/test, pass the fitted scalers from training
            dataset, loader = data_provider_with_scaler(args, flag=flag, 
                                                       scaler=train_scaler, 
                                                       target_scaler=train_target_scaler)
            logger.info(f"Created {flag} dataset with training scalers")
            
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
        "directional_accuracies": artifacts.directional_accuracies if artifacts.directional_accuracies else [],
        "best_directional_accuracy": (
            float(artifacts.best_directional_accuracy) if artifacts.best_directional_accuracy is not None else None
        ),
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
    
    # For precise loss aggregation
    total_train_loss = 0.0
    total_train_samples = 0
    
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
    for batch_index, batch_data in enumerate(train_loader):
        try:
            # 🌟 NEW: Support both legacy 4-tuple and new 6-tuple (with future celestial)
            if len(batch_data) == 6:
                # New format: includes future celestial data for deterministic conditioning
                batch_x, batch_y, batch_x_mark, batch_y_mark, future_cel_x, future_cel_mark = batch_data
                future_cel_x = future_cel_x.float().to(device)
                future_cel_mark = future_cel_mark.float().to(device)
            elif len(batch_data) == 4:
                # Legacy format: no future celestial data
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                future_cel_x, future_cel_mark = None, None
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
            
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
                amp_dtype = _resolve_amp_dtype(args)
                with torch_autocast(device_type="cuda", dtype=amp_dtype):
                    # 🌟 Pass future celestial data if available
                    if future_cel_x is not None:
                        outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 
                                          future_celestial_x=future_cel_x, 
                                          future_celestial_mark=future_cel_mark)
                    else:
                        outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                # 🌟 Pass future celestial data if available
                if future_cel_x is not None:
                    outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                      future_celestial_x=future_cel_x,
                                      future_celestial_mark=future_cel_mark)
                else:
                    outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs_tensor, aux_loss, mdn_outputs, _ = _normalize_model_output(outputs_raw)
            c_out_evaluation = len(target_indices)
            # FIXED: Extract only target features for loss computation
            batch_y_targets = batch_y[:, -args.pred_len:, :]
            if hasattr(args, 'target_wave_indices') and args.target_wave_indices:
                # Extract only the target columns
                batch_y_targets = batch_y_targets[:, :, args.target_wave_indices]
            elif hasattr(args, 'c_out') and args.c_out < batch_y_targets.shape[-1]:
                # Extract first c_out columns as targets
                batch_y_targets = batch_y_targets[:, :, :args.c_out]
            
            y_true_for_loss = scale_targets_for_loss(
                batch_y_targets, target_scaler, target_indices, device
            )
            # --- DEBUG: dump shapes and small numeric summaries to diagnostic log ---
            try:
                with open('training_diagnostic.log', 'a') as _diag_f:
                    _diag_f.write('\n--- MODEL OUTPUT DIAGNOSTICS ---\n')
                    _diag_f.write(f"outputs_tensor.shape: {tuple(outputs_tensor.shape)}\n")
                    _diag_f.write(f"outputs_tensor.mean/std: {float(outputs_tensor.mean()):.6f} / {float(outputs_tensor.std()):.6f}\n")
                    _diag_f.write(f"y_true_for_loss.shape: {tuple(y_true_for_loss.shape)}\n")
                    _diag_f.write(f"y_true_for_loss.mean/std: {float(y_true_for_loss.mean()):.6f} / {float(y_true_for_loss.std()):.6f}\n")
                    _diag_f.write(f"c_out_evaluation: {c_out_evaluation}\n")
                    enable_mdn = getattr(args, 'enable_mdn_decoder', False)
                    _diag_f.write(f"enable_mdn flag: {enable_mdn}\n")
                    if mdn_outputs is not None:
                        try:
                            pi_d, mu_d, sigma_d = mdn_outputs
                            _diag_f.write(f"mdn.pi.shape: {tuple(pi_d.shape)}\n")
                            _diag_f.write(f"mdn.mu.shape: {tuple(mu_d.shape)}\n")
                            _diag_f.write(f"mdn.sigma.shape: {tuple(sigma_d.shape)}\n")
                            # mixture mean summary
                            try:
                                w = torch.softmax(pi_d, dim=-1)
                                if mu_d.dim() == 4:
                                    # pi_d: (batch, seq, targets, components)
                                    # mu_d: (batch, seq, targets, components)  
                                    # w: (batch, seq, targets, components)
                                    # Compute weighted mean across components (last dimension)
                                    mix_mean = torch.sum(w * mu_d, dim=-1)  # (batch, seq, targets)
                                else:
                                    mix_mean = torch.sum(w * mu_d, dim=-1).unsqueeze(-1)
                                _diag_f.write(f"mdn.mixture_mean.shape: {tuple(mix_mean.shape)}\n")
                                _diag_f.write(f"mdn.mixture_mean.mean/std: {float(mix_mean.mean()):.6f} / {float(mix_mean.std()):.6f}\n")
                            except Exception as _e:
                                _diag_f.write(f"failed to compute mdn mixture mean: {_e}\n")
                        except Exception:
                            _diag_f.write("mdn_outputs present but unpacking failed\n")
                    # Also write a small slice of the outputs for visual inspection
                    try:
                        sample_pred = outputs_tensor.detach().cpu()
                        flat = sample_pred.reshape(-1)
                        sample_vals = flat[:min(8, flat.numel())].tolist()
                        _diag_f.write(f"outputs_tensor sample (first up to 8 values): {sample_vals}\n")
                    except Exception:
                        pass
            except Exception:
                pass
            # --- end debug ---

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
                # 🔍 DIAGNOSTIC: Capture weight norms before optimizer step
                if batch_index < 5 or batch_index % 50 == 0:
                    param_norms_before = {}
                    grad_norms = {}
                    for name, param in model.named_parameters():
                        if param.requires_grad and 'projection' in name:  # Track output layer
                            param_norms_before[name] = param.data.norm().item()
                            if param.grad is not None:
                                grad_norms[name] = param.grad.norm().item()
                
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # 🔍 DIAGNOSTIC: Capture weight changes after optimizer step
                if batch_index < 5 or batch_index % 50 == 0:
                    with open('training_diagnostic.log', 'a') as f:
                        f.write(f"\n--- OPTIMIZER STEP DIAGNOSTICS ---\n")
                        for name, param in model.named_parameters():
                            if param.requires_grad and 'projection' in name:
                                param_norm_after = param.data.norm().item()
                                weight_change = abs(param_norm_after - param_norms_before.get(name, 0))
                                f.write(f"{name}:\n")
                                f.write(f"  weight_norm_before: {param_norms_before.get(name, 0):.8f}\n")
                                f.write(f"  weight_norm_after: {param_norm_after:.8f}\n")
                                f.write(f"  weight_change: {weight_change:.8f}\n")
                                f.write(f"  grad_norm: {grad_norms.get(name, 0):.8f}\n")
                        f.write(f"optimizer_step_executed: {is_cycle_end or is_final_partial}\n")
                        current_lr = optimizer.param_groups[0]['lr']
                        f.write(f"current_lr: {current_lr:.8f}\n")
                
                if batch_index != total_train_batches - 1:
                    optimizer.zero_grad(set_to_none=True)

            # Accumulate raw_loss for epoch averaging
            # raw_loss is the mean loss per sample (MSELoss reduction='mean')
            # We scale it by 1/gradient_accumulation_steps for backward() to get correct gradients
            # But for reporting, we want the unscaled mean loss across batches
            train_loss += raw_loss.detach().item()
            train_batches += 1
            
            # FIXED: Sample-weighted loss aggregation for more precise training metrics
            batch_size = batch_x.size(0)
            total_train_loss += raw_loss.detach().item() * batch_size
            total_train_samples += batch_size

            # 🔍 DIAGNOSTIC: Log detailed training metrics
            if batch_index < 5 or batch_index % 50 == 0:
                with open('training_diagnostic.log', 'a') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"EPOCH {epoch+1} | BATCH {batch_index}/{total_train_batches} | TRAINING MODE\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"raw_loss (full batch loss): {raw_loss.item():.8f}\n")
                    f.write(f"loss (scaled for backward): {loss.item():.8f}\n")
                    f.write(f"effective_cycle (gradient_accumulation_steps): {effective_cycle}\n")
                    f.write(f"loss/raw_loss ratio: {loss.item()/raw_loss.item():.4f} (should be ~1/{effective_cycle})\n")
                    f.write(f"accumulated train_loss so far: {train_loss:.8f}\n")
                    f.write(f"avg train_loss so far: {train_loss / max(train_batches, 1):.8f}\n")
                    f.write(f"NOTE: NOW ACCUMULATING 'loss' (scaled), NOT 'raw_loss' (3x inflated)\n")
                    f.write(f"y_pred_for_loss.requires_grad: {y_pred_for_loss.requires_grad}\n")
                    f.write(f"y_pred_for_loss mean/std: {y_pred_for_loss.mean().item():.6f} / {y_pred_for_loss.std().item():.6f}\n")
                    f.write(f"y_true_for_loss mean/std: {y_true_for_loss.mean().item():.6f} / {y_true_for_loss.std().item():.6f}\n")
                    if aux_loss:
                        f.write(f"aux_loss contribution: {aux_loss:.8f}\n")
                    if getattr(model, "use_stochastic_learner", False):
                        f.write(f"reg_loss contribution: {(reg_loss * reg_weight).item():.8f}\n")

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
                
                # Compute detailed batch metrics including directional accuracy
                batch_metrics = _compute_detailed_batch_metrics(
                    raw_loss, outputs_tensor, y_true_for_loss, mdn_outputs, args
                )
                
                # Enhanced batch logging with directional accuracy
                dir_acc_str = ""
                if batch_metrics.get('directional_accuracy') is not None:
                    dir_acc_str = f" | dir_acc={batch_metrics['directional_accuracy']:.1f}%"
                
                logger.info(
                    "BATCH %s/%s | Epoch %s/%s | loss=%0.6f%s | Time=%0.1fs",
                    batch_index + 1,
                    len(train_loader),
                    epoch + 1,
                    args.train_epochs,
                    raw_loss.detach().item(),
                    dir_acc_str,
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

    # FIXED: Use sample-weighted average for more precise training loss
    avg_train_loss_precise = total_train_loss / max(total_train_samples, 1)
    avg_train_loss_legacy = train_loss / max(train_batches, 1)  # Keep for comparison
    
    # 🔍 DIAGNOSTIC: Log epoch summary
    with open('training_diagnostic.log', 'a') as f:
        f.write(f"\n{'#'*80}\n")
        f.write(f"EPOCH {epoch+1} TRAINING SUMMARY\n")
        f.write(f"{'#'*80}\n")
        f.write(f"total_train_loss: {train_loss:.8f}\n")
        f.write(f"train_batches: {train_batches}\n")
        f.write(f"total_train_samples: {total_train_samples}\n")
        f.write(f"avg_train_loss (legacy - batch avg): {avg_train_loss_legacy:.8f}\n")
        f.write(f"avg_train_loss (precise - sample weighted): {avg_train_loss_precise:.8f}\n")
        f.write(f"difference: {abs(avg_train_loss_precise - avg_train_loss_legacy):.8f}\n")
        f.write(f"gradient_accumulation_steps: {gradient_accumulation_steps}\n")
        f.write(f"\n")
    
    if memory_logging_enabled:
        _log_memory_snapshot(
            f"train_epoch_{epoch + 1}_end",
            device,
            MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
            {
                "avg_train_loss_precise": avg_train_loss_precise,
                "avg_train_loss_legacy": avg_train_loss_legacy, 
                "train_batches": train_batches,
                "total_train_samples": total_train_samples
            },
        )
        if getattr(args, "dump_cuda_memory_summary", False):
            _maybe_dump_cuda_summary(
                device,
                MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
                f"train_epoch_{epoch + 1}_end",
            )
    # Return the more precise sample-weighted average
    return avg_train_loss_precise, train_batches


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
) -> Tuple[float, int, Optional[float]]:
    """Run validation phase for a single epoch and return aggregate loss and directional accuracy."""

    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    # Track directional accuracy if using directional loss
    compute_dir_accuracy = False
    total_dir_accuracy = 0.0
    dir_accuracy_samples = 0
    
    try:
        from layers.modular.losses.directional_trend_loss import (
            compute_directional_accuracy,
            DirectionalTrendLoss,
            HybridMDNDirectionalLoss
        )
        compute_dir_accuracy = isinstance(criterion, (DirectionalTrendLoss, HybridMDNDirectionalLoss))
    except ImportError:
        pass

    memory_logging_enabled = getattr(args, "enable_memory_diagnostics", False)
    memory_log_interval = max(1, int(getattr(args, "memory_log_interval", 25)))
    if memory_logging_enabled:
        _log_memory_snapshot(
            f"validate_epoch_{epoch + 1}_start",
            device,
            MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
            {},
        )

    # For precise loss aggregation
    total_loss = 0.0
    total_samples = 0
    
    # Log validation start
    logger.info("Starting validation for epoch %s/%s", epoch + 1, args.train_epochs)
    
    with torch.no_grad():
        for batch_index, batch_data in enumerate(val_loader):
            try:
                # FIXED: Handle both 4-tuple and 6-tuple data formats (same as training)
                if len(batch_data) == 6:
                    # New format: includes future celestial data for deterministic conditioning
                    batch_x, batch_y, batch_x_mark, batch_y_mark, future_cel_x, future_cel_mark = batch_data
                    future_cel_x = future_cel_x.float().to(device)
                    future_cel_mark = future_cel_mark.float().to(device)
                elif len(batch_data) == 4:
                    # Legacy format: no future celestial data
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    future_cel_x, future_cel_mark = None, None
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
                
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                # 🚀 ENHANCED: Use future celestial data for validation (same as training)
                dec_inp = _create_enhanced_decoder_input(batch_y, args, logger).float().to(device)

                # FIXED: Pass future celestial data if available (consistent with training)
                if future_cel_x is not None:
                    outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                      future_celestial_x=future_cel_x,
                                      future_celestial_mark=future_cel_mark)
                else:
                    outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs_tensor, aux_loss, mdn_outputs, _ = _normalize_model_output(outputs_raw)

                c_out_evaluation = len(target_indices)
                # FIXED: Extract only target features for validation loss
                batch_y_targets = batch_y[:, -args.pred_len:, :]
                if hasattr(args, 'target_wave_indices') and args.target_wave_indices:
                    # Extract only the target columns
                    batch_y_targets = batch_y_targets[:, :, args.target_wave_indices]
                elif hasattr(args, 'c_out') and args.c_out < batch_y_targets.shape[-1]:
                    # Extract first c_out columns as targets
                    batch_y_targets = batch_y_targets[:, :, :args.c_out]
                
                y_true_for_loss = scale_targets_for_loss(
                    batch_y_targets, target_scaler, target_indices, device
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

                # FIXED: Sample-weighted loss aggregation for more precise validation metrics
                batch_size = batch_x.size(0)
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size
                
                # Keep old method for backward compatibility logging
                val_loss += float(loss.item())
                val_batches += 1
                
                # 🎯 DIRECTIONAL ACCURACY: Compute if using directional loss
                if compute_dir_accuracy:
                    try:
                        # Extract predictions for directional accuracy
                        if mdn_outputs is not None:
                            # Use MDN mean for directional calculation
                            means_da, log_stds_da, log_weights_da = mdn_outputs
                            if means_da.size(1) > args.pred_len:
                                means_da = means_da[:, -args.pred_len:, ...]
                                log_stds_da = log_stds_da[:, -args.pred_len:, ...]
                                log_weights_da = log_weights_da[:, -args.pred_len:, ...]
                            
                            # Compute mixture mean
                            weights_da = torch.softmax(log_weights_da, dim=-1)
                            if means_da.dim() == 4:
                                # weights_da: (batch, seq, targets, components)
                                # means_da: (batch, seq, targets, components)
                                # Compute weighted mean across components (last dimension)
                                pred_for_da = torch.sum(weights_da * means_da, dim=-1)  # (batch, seq, targets)
                            else:
                                pred_for_da = torch.sum(weights_da * means_da, dim=-1)
                        else:
                            pred_for_da = outputs_tensor[:, -args.pred_len:, :c_out_evaluation]
                        
                        # Compute directional accuracy
                        dir_acc = compute_directional_accuracy(pred_for_da, y_true_for_loss)
                        total_dir_accuracy += dir_acc
                        dir_accuracy_samples += 1
                    except Exception as e:
                        logger.debug(f"Could not compute directional accuracy: {e}")

                # 🔍 DIAGNOSTIC: Log validation metrics
                if batch_index < 5 or batch_index % 50 == 0:
                    with open('training_diagnostic.log', 'a') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"EPOCH {epoch+1} | BATCH {batch_index} | VALIDATION MODE\n")
                        f.write(f"{'='*80}\n")
                        f.write(f"loss: {loss.item():.8f}\n")
                        f.write(f"accumulated val_loss so far: {val_loss:.8f}\n")
                        f.write(f"avg val_loss so far: {val_loss / max(val_batches, 1):.8f}\n")
                        f.write(f"y_pred_for_loss mean/std: {y_pred_for_loss.mean().item():.6f} / {y_pred_for_loss.std().item():.6f}\n")
                        f.write(f"y_true_for_loss mean/std: {y_true_for_loss.mean().item():.6f} / {y_true_for_loss.std().item():.6f}\n")
                        if aux_loss:
                            f.write(f"aux_loss contribution: {aux_loss:.8f}\n")

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

    # FIXED: Use sample-weighted average for more precise validation loss
    avg_val_loss_precise = total_loss / max(total_samples, 1)
    avg_val_loss_legacy = val_loss / max(val_batches, 1)  # Keep for comparison
    avg_dir_accuracy = (total_dir_accuracy / max(dir_accuracy_samples, 1)) if compute_dir_accuracy else None
    
    # 🔍 DIAGNOSTIC: Log validation summary with both methods
    with open('training_diagnostic.log', 'a') as f:
        f.write(f"\n{'#'*80}\n")
        f.write(f"EPOCH {epoch+1} VALIDATION SUMMARY\n")
        f.write(f"{'#'*80}\n")
        f.write(f"total_val_loss: {val_loss:.8f}\n")
        f.write(f"val_batches: {val_batches}\n")
        f.write(f"total_samples: {total_samples}\n")
        f.write(f"avg_val_loss (legacy - batch avg): {avg_val_loss_legacy:.8f}\n")
        f.write(f"avg_val_loss (precise - sample weighted): {avg_val_loss_precise:.8f}\n")
        f.write(f"difference: {abs(avg_val_loss_precise - avg_val_loss_legacy):.8f}\n")
        if avg_dir_accuracy is not None:
            f.write(f"directional_accuracy: {avg_dir_accuracy:.2f}%\n")
        f.write(f"\n")
    
    if memory_logging_enabled:
        metrics = {
            "avg_val_loss_precise": avg_val_loss_precise,
            "avg_val_loss_legacy": avg_val_loss_legacy, 
            "val_batches": val_batches,
            "total_samples": total_samples
        }
        if avg_dir_accuracy is not None:
            metrics["directional_accuracy"] = avg_dir_accuracy
        _log_memory_snapshot(
            f"validate_epoch_{epoch + 1}_end",
            device,
            MEMORY_LOGGER if MEMORY_LOGGER.handlers else None,
            metrics,
        )
    
    # Log validation completion
    logger.info(
        "Validation complete | Epoch %s/%s | val_loss=%0.6f | batches=%s%s",
        epoch + 1,
        args.train_epochs,
        avg_val_loss_precise,
        val_batches,
        f" | dir_acc={avg_dir_accuracy:.1f}%" if avg_dir_accuracy is not None else ""
    )
    
    # Return the more precise sample-weighted average
    return avg_val_loss_precise, val_batches, avg_dir_accuracy


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
    target_scaler: Optional[Any] = None,
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
        for batch_index, batch_data in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            # FIXED: Handle both 4-tuple and 6-tuple data formats (consistent with training/validation)
            if len(batch_data) == 6:
                # New format: includes future celestial data for deterministic conditioning
                batch_x, batch_y, batch_x_mark, batch_y_mark, future_cel_x, future_cel_mark = batch_data
                future_cel_x = future_cel_x.float().to(device)
                future_cel_mark = future_cel_mark.float().to(device)
            elif len(batch_data) == 4:
                # Legacy format: no future celestial data
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                future_cel_x, future_cel_mark = None, None
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")

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

            # 🚀 ENHANCED: Use future celestial data for testing (consistent with training/validation)
            dec_inp = _create_enhanced_decoder_input(batch_y, args, logger).float().to(device)

            # FIXED: Pass future celestial data if available (consistent with training/validation)
            if future_cel_x is not None:
                outputs_raw = model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                  future_celestial_x=future_cel_x,
                                  future_celestial_mark=future_cel_mark)
            else:
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
                    # pi: [B, S, T, K], mu: [B, S, T, K]
                    # Compute weighted mean across components (last dimension)
                    pred_tensor = (pi * mu).sum(dim=-1)  # [B, S, T]
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
                    # weights_te: [B, S, T, K], means_te: [B, S, T, K]
                    # Compute weighted mean across components (last dimension)
                    pred_tensor = (weights_te * means_te).sum(dim=-1)  # [B, S, T]
                else:
                    pred_tensor = (weights_te * means_te).sum(dim=-1).unsqueeze(-1)
            else:
                pred_tensor = outputs_tensor

            # FIXED: Use scaled targets for consistent evaluation (same as training/validation)
            if target_scaler is not None:
                true_tensor = scale_targets_for_loss(
                    batch_y[:, -args.pred_len:, :], target_scaler, target_indices, device
                )
            else:
                # Fallback to unscaled targets if no scaler provided
                true_tensor = batch_y[:, -args.pred_len:, :]
                if target_indices:
                    true_tensor = true_tensor[:, :, target_indices]

            pred_aligned = pred_tensor[:, -args.pred_len:, :]
            true_aligned = true_tensor
            if target_indices and target_scaler is None:
                # Only slice predictions if we didn't already slice targets via scaler
                pred_aligned = pred_aligned[:, :, target_indices]

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
    target_scaler: Optional[Any] = None,
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
        target_scaler=target_scaler,
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
            target_scaler=target_scaler,
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

def train_celestial_pgat_production(config_path: str = "configs/celestial_enhanced_pgat_production.yaml") -> bool:
    """Production training function.

    When invoked from CLI, an alternate YAML config path can be supplied via
    the ``--config`` argument. This function keeps a backward-compatible
    default pointing to the production configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        True if training and evaluation succeed, else False.
    """
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config_dict = yaml.safe_load(config_file)

    args = SimpleConfig(config_dict)
    logger = configure_logging(args)

    # 🔍 DIAGNOSTIC: Initialize diagnostic log
    with open('training_diagnostic.log', 'w') as f:
        f.write(f"CELESTIAL ENHANCED PGAT - TRAINING DIAGNOSTICS\n")
        f.write(f"{'='*80}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
    
    logger.info("Starting PRODUCTION Celestial Enhanced PGAT training run")
    logger.info("Heavy-duty overnight configuration enabled")
    logger.info("🔍 DIAGNOSTIC MODE ENABLED - Writing to training_diagnostic.log")
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

    # Docker shared memory advisory to prevent dataloader stalls
    try:
        import os
        import math
        shm = os.statvfs('/dev/shm')
        shm_bytes = shm.f_frsize * shm.f_blocks
        shm_gb = shm_bytes / (1024 ** 3)
        pin_memory = bool(getattr(args, 'pin_memory', True)) and torch.cuda.is_available()
        if shm_gb < 1.0 and (getattr(args, 'num_workers', 0) > 0 or pin_memory):
            logger.warning(
                "Low /dev/shm (%.2f GiB) detected. To avoid DataLoader stalls in Docker, either: "
                "use --shm-size=8g or --ipc=host, or set num_workers=0 and pin_memory=false.",
                shm_gb,
            )
    except Exception:
        pass

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

    logger.info("Validating configuration for dimension compatibility...")
    try:
        # Validate configuration BEFORE model initialization
        Model.validate_configuration(args)
        logger.info("✅ Configuration validation passed")
    except ValueError as config_error:
        logger.error("❌ Configuration validation failed:")
        logger.error(str(config_error))
        return False
    
    logger.info("Validating configuration for dimension compatibility...")
    try:
        # 🚀 NEW: Validate configuration BEFORE model initialization
        Model.validate_configuration(args)
        logger.info("✅ Configuration validation passed")
    except ValueError as config_error:
        logger.error("❌ Configuration validation failed:")
        logger.error(str(config_error))
        return False
    
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
    
    try:
        from layers.modular.losses.directional_trend_loss import (
            DirectionalTrendLoss,
            HybridMDNDirectionalLoss
        )
    except ImportError:
        DirectionalTrendLoss = None
        HybridMDNDirectionalLoss = None
        logger.warning("DirectionalTrendLoss not available - install from layers.modular.losses")
    
    # Use proper loss function based on config
    loss_config = getattr(args, 'loss', {})
    loss_type = loss_config.get('type', 'auto') if isinstance(loss_config, dict) else 'auto'
    
    if loss_type == 'hybrid_mdn_directional' and HybridMDNDirectionalLoss is not None:
        # Hybrid MDN + Directional loss for directional/trend-focused forecasting
        nll_weight = loss_config.get('nll_weight', 0.3)
        direction_weight = loss_config.get('direction_weight', 3.0)
        trend_weight = loss_config.get('trend_weight', 1.5)
        magnitude_weight = loss_config.get('magnitude_weight', 0.1)
        
        criterion = HybridMDNDirectionalLoss(
            nll_weight=nll_weight,
            direction_weight=direction_weight,
            trend_weight=trend_weight,
            magnitude_weight=magnitude_weight
        )
        logger.info(
            "Using Hybrid MDN+Directional Loss | nll_weight=%.2f direction_weight=%.2f trend_weight=%.2f magnitude_weight=%.2f",
            nll_weight, direction_weight, trend_weight, magnitude_weight
        )
    elif loss_type == 'directional_trend' and DirectionalTrendLoss is not None:
        # Pure directional/trend loss (no MDN uncertainty)
        direction_weight = loss_config.get('direction_weight', 5.0)
        trend_weight = loss_config.get('trend_weight', 2.0)
        magnitude_weight = loss_config.get('magnitude_weight', 0.1)
        
        criterion = DirectionalTrendLoss(
            direction_weight=direction_weight,
            trend_weight=trend_weight,
            magnitude_weight=magnitude_weight,
            use_mdn_mean=getattr(model, 'use_mixture_decoder', False)
        )
        logger.info(
            "Using Directional Trend Loss | direction_weight=%.2f trend_weight=%.2f magnitude_weight=%.2f",
            direction_weight, trend_weight, magnitude_weight
        )
    elif getattr(model, 'use_mixture_decoder', False) and SequentialMixtureNLLLoss is not None:
        # Auto-select mixture loss for MDN models
        criterion = SequentialMixtureNLLLoss(reduction='mean')
        logger.info("Using Gaussian Mixture NLL Loss for probabilistic predictions")
    else:
        # Fallback to MSE
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

        # Adjust learning rate BEFORE training this epoch
        current_lr = adjust_learning_rate_warmup_cosine(optimizer, epoch, args)

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

        avg_val_loss, val_batches, avg_dir_accuracy = validate_epoch(
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

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - training_start_time
        remaining_epochs = args.train_epochs - (epoch + 1)
        estimated_remaining = (total_elapsed / (epoch + 1) * remaining_epochs) if (epoch + 1) else 0.0

        warmup_status = f" [WARMUP {epoch+1}/{warmup_epochs}]" if epoch < warmup_epochs else ""
        dir_acc_str = f" dir_acc={avg_dir_accuracy:.2f}%" if avg_dir_accuracy is not None else ""
        
        # Enhanced epoch summary logging
        logger.info("=" * 100)
        logger.info(
            "🎯 EPOCH %s/%s COMPLETE%s | ⏱️  %0.2fs | 🔥 %0.2fh elapsed | ⏳ %0.2fh remaining",
            epoch + 1,
            args.train_epochs,
            warmup_status,
            epoch_time,
            total_elapsed / 3600,
            estimated_remaining / 3600 if estimated_remaining else 0.0,
        )
        logger.info(
            "📊 LOSSES: Train=%0.6f | Val=%0.6f | Δ=%+0.6f | LR=%0.8f",
            avg_train_loss,
            avg_val_loss,
            avg_val_loss - avg_train_loss,
            current_lr,
        )
        if avg_dir_accuracy is not None:
            logger.info(
                "🎯 DIRECTIONAL ACCURACY: %0.2f%% | Trend Quality: %s",
                avg_dir_accuracy,
                "Excellent" if avg_dir_accuracy > 60 else "Good" if avg_dir_accuracy > 50 else "Improving"
            )
        
        # Performance indicators
        if epoch > 0:
            prev_val_loss = artifacts.val_losses[-1] if artifacts.val_losses else float('inf')
            improvement = prev_val_loss - avg_val_loss
            trend = "📈 Improving" if improvement > 0.001 else "📉 Declining" if improvement < -0.001 else "➡️  Stable"
            logger.info(f"📈 TREND: {trend} | Improvement: {improvement:+0.6f}")
        
        logger.info("=" * 100)

        if train_batches == 0:
            logger.warning("Epoch %s processed zero training batches; check data loader configuration.", epoch + 1)
        if val_batches == 0:
            logger.warning("Epoch %s processed zero validation batches; check data loader configuration.", epoch + 1)

        is_best = artifacts.record_epoch(epoch, avg_train_loss, avg_val_loss, avg_dir_accuracy)
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
        target_scaler=target_scaler,
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

def main() -> int:
    """CLI entrypoint.

    Supports an optional ``--config`` argument to override the default
    production YAML path. Any unknown extra arguments are ignored to keep
    compatibility with external runners.
    """
    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=str, default="configs/celestial_enhanced_pgat_production.yaml")
        # Parse only known args to avoid breaking external wrappers
        cli_args, _ = parser.parse_known_args()

        success = train_celestial_pgat_production(config_path=cli_args.config)
        return 0 if success else 1
    except Exception as exc:
        LOGGER.exception("PRODUCTION training failed with error: %s", exc)
        return 1

if __name__ == "__main__":
    exit(main())