"""Dynamic training entrypoint for the SOTA Temporal PGAT architecture.

This script mirrors the robust data-pipeline integration used by
``train_dynamic_autoformer.py`` while configuring arguments that the
`SOTA_Temporal_PGAT` model expects (dynamic graphs, autocorrelation attention,
and Mixture Density decoding).  It is intended to bridge legacy YAML configs
with the modern experiment runner (`Exp_Long_Term_Forecast`).

Usage
-----
    python scripts/train/train_dynamic_pgat.py --config path/to/pgat.yaml

The referenced YAML file should contain both data settings (paths, targets,
batch size, etc.) and high-level model hyperparameters.  See the repository
documentation in ``docs/SOTA_PGAT_Documentation.md`` for guidance.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from data_provider.data_factory import setup_financial_forecasting_data
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def _ensure_attr(args: SimpleNamespace, name: str, default: Any) -> None:
    """Populate ``args`` with ``default`` if the attribute ``name`` is missing."""

    if not hasattr(args, name):
        setattr(args, name, default)


def _configure_pgat_args(args: SimpleNamespace) -> None:
    """Set PGAT-specific defaults expected by ``SOTA_Temporal_PGAT``.

    Parameters
    ----------
    args:
        The mutable configuration namespace constructed from the YAML file.
    """

    # Use model from config if specified, otherwise default to SOTA_Temporal_PGAT
    if not hasattr(args, 'model') or args.model is None:
        args.model = "SOTA_Temporal_PGAT"
    _ensure_attr(args, "model_type", "pgat")
    _ensure_attr(args, "model_id", "pgat_dynamic")

    # Core architectural toggles
    _ensure_attr(args, "use_dynamic_edge_weights", True)
    _ensure_attr(args, "enable_graph_attention", True)
    _ensure_attr(args, "enable_dynamic_graph", True)
    _ensure_attr(args, "enable_graph_positional_encoding", True)
    _ensure_attr(args, "enable_structural_pos_encoding", True)
    _ensure_attr(args, "use_autocorr_attention", True)
    _ensure_attr(args, "use_adaptive_temporal", True)
    _ensure_attr(args, "use_mixture_density", False)
    _ensure_attr(args, "mdn_components", 3)

    # Graph hyper-parameters
    _ensure_attr(args, "graph_dim", args.d_model)
    _ensure_attr(args, "graph_alpha", 0.2)
    _ensure_attr(args, "graph_beta", 0.1)
    _ensure_attr(args, "num_nodes", getattr(args, "enc_in", args.d_model))

    # Training knobs
    _ensure_attr(args, "loss", "mse")
    _ensure_attr(args, "loss_function_type", "mse")
    _ensure_attr(args, "learning_rate", 1e-4)
    _ensure_attr(args, "train_epochs", 20)
    _ensure_attr(args, "patience", 5)
    _ensure_attr(args, "batch_size", 32)
    _ensure_attr(args, "num_workers", 4)


def _setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logging handlers for the script."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    return log


def _load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration with lazy import of PyYAML."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    yaml = importlib.import_module("yaml")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _set_common_defaults(args: SimpleNamespace) -> None:
    """Populate generic training defaults shared across models."""

    _ensure_attr(args, "use_amp", False)
    _ensure_attr(args, "use_multi_gpu", False)
    _ensure_attr(args, "devices", "0")
    _ensure_attr(args, "quantile_levels", [])
    _ensure_attr(args, "embed", "timeF")
    _ensure_attr(args, "activation", "gelu")
    _ensure_attr(args, "output_attention", False)
    _ensure_attr(args, "scale", True)
    _ensure_attr(args, "task_name", "long_term_forecast")
    _ensure_attr(args, "freq", "h")

    _ensure_attr(args, "seq_len", 96)
    _ensure_attr(args, "label_len", 48)
    _ensure_attr(args, "pred_len", 24)
    _ensure_attr(args, "d_model", 512)
    _ensure_attr(args, "n_heads", 8)
    _ensure_attr(args, "d_ff", 2048)
    _ensure_attr(args, "dropout", 0.1)
    _ensure_attr(args, "factor", 1)
    _ensure_attr(args, "features", "M")


def _auto_select_device(args: SimpleNamespace, log: logging.Logger) -> None:
    """Set ``use_gpu``/``gpu_type`` flags based on hardware availability."""

    if not hasattr(args, "use_gpu"):
        if torch.cuda.is_available():
            args.use_gpu = True
            args.gpu_type = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.use_gpu = True
            args.gpu_type = "mps"
        else:
            args.use_gpu = False
            args.gpu_type = "cpu"
        log.info("Hardware detection -> use_gpu=%s gpu_type=%s", args.use_gpu, args.gpu_type)


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Train SOTA Temporal PGAT with dynamic pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sota_pgat_dynamic.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose DEBUG logging output.",
    )

    cli_args = parser.parse_args()
    log = _setup_logging(verbose=cli_args.verbose)

    try:
        raw_config = _load_config(cli_args.config)
        log.info("Loaded configuration from %s", cli_args.config)

        args = SimpleNamespace(**raw_config)
        args.config = cli_args.config

        _set_common_defaults(args)
        _configure_pgat_args(args)
        _auto_select_device(args, log)

        # Ensure dataset configuration essentials exist
        _ensure_attr(args, "root_path", os.path.join(PROJECT_ROOT, "data"))
        _ensure_attr(args, "data_path", "financial.csv")
        _ensure_attr(args, "target", "Close")

        min_required = args.seq_len + args.pred_len
        _ensure_attr(args, "validation_length", max(150, min_required))
        _ensure_attr(args, "test_length", max(120, min_required))

        log.info("Setting up financial forecasting data pipeline...")
        train_loader, vali_loader, test_loader, scaler_manager, dim_manager = setup_financial_forecasting_data(args)

        args.train_loader = train_loader
        args.vali_loader = vali_loader
        args.test_loader = test_loader
        args.scaler_manager = scaler_manager
        args.dim_manager = dim_manager

        args.enc_in = getattr(args, "enc_in", dim_manager.enc_in)
        args.dec_in = getattr(args, "dec_in", dim_manager.dec_in)
        args.c_out = dim_manager.c_out_model
        args.c_out_evaluation = getattr(args, "c_out_evaluation", dim_manager.c_out_evaluation)

        log.debug(
            "DimensionManager -> enc_in=%s dec_in=%s c_out_model=%s c_out_eval=%s",
            dim_manager.enc_in,
            dim_manager.dec_in,
            dim_manager.c_out_model,
            dim_manager.c_out_evaluation,
        )

        experiment = Exp_Long_Term_Forecast(args)
        log.info("Experiment instantiated successfully")

        setting_id = f"{args.model_id}_{args.features}_{args.seq_len}_{args.pred_len}"
        log.info("Training run identifier: %s", setting_id)

        experiment.train(setting_id)
        log.info("Training completed; starting evaluation")
        experiment.test(setting_id, test=1)
        log.info("PGAT dynamic training pipeline finished successfully")

    except Exception as exc:  # pylint: disable=broad-except
        log.error("Unhandled exception during PGAT training: %s", exc)
        traceback.print_exc()


if __name__ == "__main__":
    main()