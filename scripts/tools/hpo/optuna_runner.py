"""
Optuna-based hyperparameter optimization runner for Celestial Enhanced PGAT.

This runner works without modifying your training code by:
- Cloning a base YAML config per trial with hyperparameter overrides
- Running the production trainer as a subprocess
- Reading best_val_loss from checkpoints/<model_id>/production_results.json

Requirements:
- pip install optuna pyyaml
- Optional: pip install optuna-dashboard

Usage (examples):
  python -X utf8 tools/hpo/optuna_runner.py \
      --base_config configs/celestial_enhanced_pgat_production.yaml \
      --trials 10 --epochs 3 --storage hpo_runs/db.sqlite3

Afterwards (optional):
  optuna-dashboard sqlite:///hpo_runs/db.sqlite3

Notes:
- Trials run sequentially by default (safe for a single GPU). Increase parallelism by
  running multiple processes (or use Ray Tune for distributed sweeps).
- The objective minimized is best_val_loss found during training.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
import yaml


@dataclass
class HPOConfig:
    base_config: Path
    trials: int
    epochs: int
    storage: Optional[str]
    study_name: str
    sampler: str
    seed: int
    work_dir: Path
    python_exe: str


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _build_trial_config(base_cfg: Dict[str, Any], trial: optuna.Trial, epochs: int, model_id: str) -> Dict[str, Any]:
    cfg = dict(base_cfg) if isinstance(base_cfg, dict) else {}

    # Top-level simple keys used by the trainer (see train_celestial_production.py)
    # Fall back to setting even if missing in base YAML — trainer reads via getattr with defaults
    cfg["model_id"] = model_id
    cfg["train_epochs"] = int(epochs)

    # Search space (conservative, GPU-friendly)
    cfg["learning_rate"] = float(trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True))
    cfg["weight_decay"] = float(trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
    cfg["batch_size"] = int(trial.suggest_categorical("batch_size", [4, 8, 12, 16]))
    cfg["gradient_accumulation_steps"] = int(trial.suggest_categorical("grad_accum", [1, 2, 4]))
    cfg["warmup_epochs"] = int(trial.suggest_int("warmup_epochs", 0, max(1, epochs // 3)))

    # Light architectural knobs (keep safe ranges to avoid OOM):
    cfg["n_heads"] = int(trial.suggest_categorical("n_heads", [4, 8]))
    cfg["e_layers"] = int(trial.suggest_int("e_layers", 2, 4))
    cfg["d_layers"] = int(trial.suggest_int("d_layers", 1, 2))
    
    # Directional loss weights (if using directional/hybrid loss)
    # Check if base config uses directional loss
    if "loss" in cfg and isinstance(cfg["loss"], dict):
        loss_type = cfg["loss"].get("type", "auto")
        if loss_type in ["directional_trend", "hybrid_mdn_directional"]:
            if "direction_weight" not in cfg["loss"]:
                cfg["loss"]["direction_weight"] = float(trial.suggest_float("direction_weight", 1.0, 10.0, log=True))
            if "trend_weight" not in cfg["loss"]:
                cfg["loss"]["trend_weight"] = float(trial.suggest_float("trend_weight", 0.5, 5.0, log=True))
            if "magnitude_weight" not in cfg["loss"]:
                cfg["loss"]["magnitude_weight"] = float(trial.suggest_float("magnitude_weight", 0.01, 1.0, log=True))
            # For hybrid loss, tune the NLL weight as well
            if loss_type == "hybrid_mdn_directional" and "nll_weight" not in cfg["loss"]:
                cfg["loss"]["nll_weight"] = float(trial.suggest_float("nll_weight", 0.1, 0.5))

    # Optional diagnostics toggles (keep disabled for faster HPO by default)
    cfg.setdefault("enable_gradient_diagnostics", False)
    cfg.setdefault("enable_attention_logging", False)
    cfg.setdefault("enable_component_profiling", False)

    return cfg



def _run_training(python_exe: str, trial_config_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    cmd = [python_exe, "-X", "utf8", "scripts/train/train_celestial_production.py", "--config", str(trial_config_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    # Stream output to console for visibility
    sys.stdout.write(proc.stdout.decode("utf-8", errors="ignore"))
    sys.stdout.flush()
    return proc.returncode


def _read_metric(model_id: str) -> float:
    results_path = Path("checkpoints") / model_id / "production_results.json"
    if not results_path.exists():
        # Penalize missing metrics strongly so Optuna avoids unstable configs
        raise FileNotFoundError(f"Missing results file: {results_path}")
    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    best = data.get("best_val_loss")
    if best is None or not isinstance(best, (float, int)):
        raise ValueError(f"Invalid best_val_loss in {results_path}: {best}")
    return float(best)


def build_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    name = name.lower()
    if name in ("tpe", "default"):
        return optuna.samplers.TPESampler(seed=seed)
    if name == "cmaes":
        try:
            from optuna.samplers import CMAESampler  # type: ignore
            return CMAESampler(seed=seed)
        except Exception:
            return optuna.samplers.TPESampler(seed=seed)
    if name == "gpo" or name == "gpsampler":
        try:
            from optuna.samplers import GPSampler  # type: ignore
            return GPSampler(seed=seed)
        except Exception:
            return optuna.samplers.TPESampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Optuna HPO runner for Celestial Enhanced PGAT")
    parser.add_argument("--base_config", type=str, required=True, help="Path to base YAML config")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=3, help="Train epochs per trial (kept small for speed)")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URI, e.g., sqlite:///hpo_runs/db.sqlite3")
    parser.add_argument("--study_name", type=str, default="celestial_pgat_hpo", help="Optuna study name")
    parser.add_argument("--sampler", type=str, default="tpe", help="Sampler: tpe|cmaes|gpsampler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampler")
    parser.add_argument("--work_dir", type=str, default="hpo_runs", help="Working directory for HPO artifacts")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to run training")

    args = parser.parse_args()

    cfg = HPOConfig(
        base_config=Path(args.base_config),
        trials=args.trials,
        epochs=args.epochs,
        storage=args.storage,
        study_name=args.study_name,
        sampler=args.sampler,
        seed=args.seed,
        work_dir=Path(args.work_dir),
        python_exe=args.python,
    )

    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    (cfg.work_dir / "configs").mkdir(parents=True, exist_ok=True)

    base_yaml = _load_yaml(cfg.base_config)

    # Prepare Optuna study
    sampler = build_sampler(cfg.sampler, cfg.seed)
    storage = cfg.storage
    if storage:
        # Ensure local sqlite path directory exists
        if storage.startswith("sqlite:///"):
            db_path = storage.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(direction="minimize", study_name=cfg.study_name, sampler=sampler, storage=storage, load_if_exists=True)
    else:
        study = optuna.create_study(direction="minimize", study_name=cfg.study_name, sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"hpo_{trial.number}_{ts}"

        # Create per-trial YAML
        trial_yaml = _build_trial_config(base_yaml, trial, cfg.epochs, model_id)
        trial_cfg_path = cfg.work_dir / "configs" / f"trial_{trial.number:04d}.yaml"
        _dump_yaml(trial_cfg_path, trial_yaml)

        # Run training
        start = time.time()
        rc = _run_training(cfg.python_exe, trial_cfg_path)
        duration = time.time() - start

        # Attach metadata
        trial.set_user_attr("duration_sec", duration)
        trial.set_user_attr("model_id", model_id)
        trial.set_user_attr("config_path", str(trial_cfg_path))

        if rc != 0:
            # Failed run — mark as failed by raising
            raise RuntimeError(f"Training subprocess exited with code {rc}")

        # Read metric
        score = _read_metric(model_id)
        # Report intermediate value for future pruning integration
        trial.report(score, step=cfg.epochs)
        return score

    study.optimize(objective, n_trials=cfg.trials)

    best = study.best_trial
    print("\n===== HPO COMPLETE =====")
    print(f"Study: {cfg.study_name}")
    print(f"Best value (best_val_loss): {best.value:.6f}")
    print(f"Best params: {best.params}")
    print(f"Trial attrs: {{'model_id': {best.user_attrs.get('model_id')}, 'duration_sec': {best.user_attrs.get('duration_sec')}}}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
