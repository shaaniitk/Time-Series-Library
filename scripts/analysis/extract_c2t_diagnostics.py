#!/usr/bin/env python3
"""
Extract and visualize Celestial→Target attention diagnostics.

- Loads minimal diagnostic config (configs/celestial_diagnostic_minimal.yaml)
- Builds the model and a single batch from the training dataloader
- Runs one forward pass to capture `celestial_target_diagnostics` from metadata
- Aggregates per-target attention over batch and pred_len
- Saves:
  - JSON snapshot: logs/c2t_attention_snapshot.json
  - Heatmap PNG:   logs/c2t_attention_heatmap.png (if matplotlib available)

Usage:
  python -X utf8 scripts/analysis/extract_c2t_diagnostics.py

Notes:
- This script is read-only w.r.t. model weights. It does not train.
- If matplotlib is not available, the JSON is still generated.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import yaml  # type: ignore

from scripts.train.train_celestial_production import (
    SimpleConfig,
    prepare_data_modules,
)
from models.Celestial_Enhanced_PGAT import Model

LOGGER = logging.getLogger("analysis.c2t")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

CONFIG_PATH = PROJECT_ROOT / "configs" / "celestial_diagnostic_minimal.yaml"
OUTPUT_JSON = PROJECT_ROOT / "logs" / "c2t_attention_snapshot.json"
OUTPUT_PNG = PROJECT_ROOT / "logs" / "c2t_attention_heatmap.png"


def _load_config(config_path: Path) -> SimpleConfig:
    """Load YAML into SimpleConfig with minimal overrides for a quick single-batch run."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    # Force minimal loader size: we only need one batch
    cfg_dict.setdefault("train_epochs", 1)
    cfg_dict.setdefault("batch_size", 8)
    cfg_dict.setdefault("num_workers", 0)
    cfg_dict.setdefault("use_gpu", torch.cuda.is_available())
    cfg_dict.setdefault("collect_diagnostics", True)
    cfg_dict.setdefault("verbose_logging", True)
    # Ensure C→T diagnostics are enabled
    cfg_dict.setdefault("use_celestial_target_attention", True)
    cfg_dict.setdefault("celestial_target_diagnostics", True)
    # Legacy data_provider expects these
    cfg_dict.setdefault("task_name", "long_term_forecast")
    cfg_dict.setdefault("model_name", cfg_dict.get("model", "Celestial_Enhanced_PGAT"))
    cfg_dict.setdefault("data", cfg_dict.get("data", "custom"))
    return SimpleConfig(cfg_dict)


def _get_device(args: SimpleConfig) -> torch.device:
    """Return best-available device honoring args.use_gpu."""
    if bool(getattr(args, "use_gpu", True)) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _run_forward_single_batch(model: Model, batch: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    """Run a forward pass on a single batch and return the metadata dict."""
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    device = next(model.parameters()).device

    # Build decoder input the same way as training: concat last label_len with future celestial window
    from scripts.train.train_celestial_production import _create_enhanced_decoder_input
    dec_inp = _create_enhanced_decoder_input(batch_y, model.configs, LOGGER).float().to(device)

    outputs_raw = model(batch_x.float().to(device), batch_x_mark.float().to(device), dec_inp, batch_y_mark.float().to(device))

    # Normalize to (preds, metadata) or (preds, aux, mdn, metadata)
    from scripts.train.train_celestial_production import _normalize_model_output
    _, _, _, metadata = _normalize_model_output(outputs_raw)

    if metadata is None:
        return {}
    return metadata


def _aggregate_attention(diagnostics: Dict[str, Any]) -> Tuple[List[List[float]], List[str]]:
    """Aggregate per-target attention over [batch, pred_len] to [num_targets x num_celestial].

    Returns:
      - matrix: list[num_targets][num_celestial] of mean attention weights
      - target_labels: list of target names inferred from config or indices
    """
    attn_dict = diagnostics.get("attention_weights", {})
    if not attn_dict:
        return [], []

    # keys look like 'target_0_attn', each tensor [batch, pred_len, num_celestial]
    per_target: List[List[float]] = []
    target_indices: List[int] = []
    for key in sorted(attn_dict.keys()):
        if not key.startswith("target_") or not key.endswith("_attn"):
            continue
        try:
            idx = int(key.split("_")[1])
        except Exception:
            continue
        tensor = attn_dict[key]
        if isinstance(tensor, torch.Tensor):
            t = tensor
        else:
            # If already moved to CPU numpy-like via model._move_to_cpu
            t = torch.as_tensor(tensor)
        mean_vec = t.mean(dim=(0, 1))  # [num_celestial]
        per_target.append(mean_vec.detach().cpu().tolist())
        target_indices.append(idx)

    # Sort by target index just in case
    order = sorted(range(len(target_indices)), key=lambda i: target_indices[i])
    per_target_sorted = [per_target[i] for i in order]
    # Generate labels (OHLC common case)
    target_labels = [f"target_{i}" for i in sorted(target_indices)]
    return per_target_sorted, target_labels


def _save_json(matrix: List[List[float]], target_labels: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"targets": target_labels, "attention": matrix}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    LOGGER.info("Saved C→T attention snapshot JSON to %s", out_path)


def _save_heatmap(matrix: List[List[float]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.array(matrix)  # [num_targets, num_celestial]
        plt.figure(figsize=(10, 4))
        im = plt.imshow(arr, aspect="auto", cmap="viridis")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Celestial body index")
        plt.ylabel("Target index")
        plt.title("Celestial→Target attention (mean over batch,time)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        LOGGER.info("Saved C→T attention heatmap to %s", out_path)
    except Exception as exc:  # pragma: no cover - plotting optional
        LOGGER.warning("Heatmap generation skipped: %s", exc)


def main() -> None:
    # Load config and data
    args = _load_config(CONFIG_PATH)
    device = _get_device(args)

    modules = prepare_data_modules(args, LOGGER, device)
    train_loader = modules["train"].loader
    if len(train_loader) == 0:
        raise RuntimeError("Train loader has zero batches; check dataset configuration")

    # Initialize model
    model = Model(args).to(device)
    model.eval()

    # Take a single batch
    batch = next(iter(train_loader))

    # Forward pass to collect diagnostics
    metadata = _run_forward_single_batch(model, batch)
    c2t = metadata.get("celestial_target_diagnostics", {}) if isinstance(metadata, dict) else {}
    if not c2t:
        LOGGER.warning("No celestial_target_diagnostics found. Ensure flags are enabled and forward executed.")
        return

    matrix, target_labels = _aggregate_attention(c2t)
    if not matrix:
        LOGGER.warning("Attention diagnostics present but empty. Skipping outputs.")
        return

    _save_json(matrix, target_labels, OUTPUT_JSON)
    _save_heatmap(matrix, OUTPUT_PNG)


if __name__ == "__main__":
    main()
