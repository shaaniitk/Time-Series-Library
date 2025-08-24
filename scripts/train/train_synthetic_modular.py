import os
import sys
import math
import argparse
import traceback
from types import SimpleNamespace
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

# Ensure project root on path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_provider.data_factory import setup_financial_forecasting_data
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
try:
    from utils.logger import logger
except Exception:  # fallback basic logger
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("synthetic_modular")

# Reuse component configuration helper
from scripts.train.train_dynamic_autoformer import _configure_modular_autoformer_args


def generate_sine_dataframe(length: int, start_date: str = '2023-01-01', freq: str = 'D',
                            kind: str = 'sine', base_freq: float = 1.0, amplitude: float = 1.0,
                            phase: float = 0.0, noise_std: float = 0.0) -> pd.DataFrame:
    """Generate a synthetic univariate sine/cosine series with a date column.

    Columns: ['date', 'y']
    """
    # Build date index
    start_dt = pd.to_datetime(start_date)
    if freq.upper() == 'B':
        dates = pd.bdate_range(start=start_dt, periods=length)
    else:
        dates = pd.date_range(start=start_dt, periods=length, freq=freq.upper())

    # Generate signal
    t = np.linspace(0.0, 2 * math.pi, num=length, endpoint=False)
    signal = amplitude * (np.sin(base_freq * t + phase) if kind == 'sine' else np.cos(base_freq * t + phase))
    if noise_std > 0:
        signal = signal + np.random.normal(0, noise_std, size=length)

    df = pd.DataFrame({
        'date': dates,
        'y': signal.astype(float),
    })
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description='Train Modular Autoformer on synthetic sine/cosine data (YAML-driven)')
    parser.add_argument('--config', type=str, default='configs/modular_enhanced_light.yaml', help='Path to YAML config')
    parser.add_argument('--length', type=int, default=400, help='Total number of synthetic rows')
    parser.add_argument('--kind', type=str, default='sine', choices=['sine', 'cosine'])
    parser.add_argument('--base-freq', type=float, default=1.0)
    parser.add_argument('--amplitude', type=float, default=1.0)
    parser.add_argument('--phase', type=float, default=0.0)
    parser.add_argument('--noise-std', type=float, default=0.0)
    args_cli = parser.parse_args()

    try:
        # Load YAML into a SimpleNamespace like train_dynamic_autoformer
        import importlib
        yaml = importlib.import_module('yaml')
        with open(args_cli.config, 'r') as f:
            cfg = yaml.safe_load(f)
        args = SimpleNamespace(**cfg)

        # Defaults matching train_dynamic_autoformer
        for k, v in [
            ('use_amp', False), ('augmentation_ratio', 0), ('use_multi_gpu', False),
            ('devices', '0'), ('quantile_levels', []), ('embed', 'timeF'),
            ('activation', 'gelu'), ('output_attention', False), ('scale', True),
            ('task_name', 'long_term_forecast'), ('d_model', 512), ('freq', args.freq if hasattr(args, 'freq') else 'D'),
            ('dropout', 0.1), ('factor', 1), ('n_heads', 8), ('d_ff', 2048), ('lradj', 'type1'),
            ('model_id', 'modular_sine'), ('features', 'S'),
        ]:
            if not hasattr(args, k):
                setattr(args, k, v)

        # Hardware hints
        if not hasattr(args, 'use_gpu'):
            args.use_gpu = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
        if not hasattr(args, 'gpu_type'):
            if torch.cuda.is_available():
                args.gpu_type = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                args.gpu_type = 'mps'
            else:
                args.gpu_type = 'cpu'

        # Ensure validation/test lengths are big enough for windows
        min_len = args.seq_len + args.pred_len
        args.validation_length = max(getattr(args, 'validation_length', min_len), min_len)
        args.test_length = max(getattr(args, 'test_length', min_len), min_len)

        # Runtime/data defaults expected by the data factory
        if not hasattr(args, 'num_workers'):
            args.num_workers = 0
        if not hasattr(args, 'loss'):
            args.loss = 'mse'
        # Type coercions for safety
        if hasattr(args, 'learning_rate'):
            try:
                args.learning_rate = float(args.learning_rate)
            except Exception:
                args.learning_rate = 1e-3
        else:
            args.learning_rate = 1e-3

        # Generate synthetic dataset and save to a temporary CSV under data/
        os.makedirs(os.path.join(project_root, 'data'), exist_ok=True)
        tmp_csv = os.path.join(project_root, 'data', 'tmp_sine.csv')
        df = generate_sine_dataframe(
            length=args_cli.length,
            start_date='2023-01-01',
            freq=getattr(args, 'freq', 'D'),
            kind=args_cli.kind,
            base_freq=args_cli.base_freq,
            amplitude=args_cli.amplitude,
            phase=args_cli.phase,
            noise_std=args_cli.noise_std,
        )
        df.to_csv(tmp_csv, index=False)

        # Wire args to use the synthetic CSV
        args.root_path = 'data'
        args.data_path = os.path.basename(tmp_csv)
        args.target = getattr(args, 'target', 'y')
        if not hasattr(args, 'checkpoints'):
            args.checkpoints = './checkpoints/'

        logger.info(f"Synthetic data written to: {tmp_csv} with shape {df.shape}")

        # Data pipeline
        train_loader, vali_loader, test_loader, scaler_manager, dim_manager = setup_financial_forecasting_data(args)

        # Pass managers/loaders to args as expected by Exp
        args.train_loader = train_loader
        args.vali_loader = vali_loader
        args.test_loader = test_loader
        args.scaler_manager = scaler_manager
        args.dim_manager = dim_manager

        # Model IO dims
        args.enc_in = getattr(args, 'enc_in', dim_manager.enc_in)
        args.dec_in = getattr(args, 'dec_in', dim_manager.dec_in)
        # Always use model c_out from DimensionManager (handles quantiles)
        args.c_out = dim_manager.c_out_model
        args.c_out_evaluation = getattr(args, 'c_out_evaluation', dim_manager.c_out_evaluation)

        # Select components for ModularAutoformer
        _configure_modular_autoformer_args(args)

        # Run experiment
        exp = Exp_Long_Term_Forecast(args)
        setting = f"{args.model_id}_{args.features}_{args.seq_len}_{args.pred_len}"
        logger.info("Starting training on synthetic data")
        exp.train(setting)
        logger.info("Starting testing on synthetic data")
        exp.test(setting, test=1)

    except Exception as e:
        logger.error(f"Error in synthetic modular training: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
