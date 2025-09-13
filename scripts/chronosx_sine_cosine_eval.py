"""ChronosXAutoformer sine & cosine forecasting evaluation.

Enhancements (recommendations 1-5 implemented):
 1. Multi-period contexts: specify --periods (default 1.0) to cover more full cycles.
 2. Normalization comparison: use --compare-norm to evaluate none,standard,minmax in one run.
 3. Uncertainty sampling: set --uncertainty and --num-samples to average multiple samples.
 4. Lightweight calibration: optional residual linear calibrator (--calibrate-steps) trained
        on synthetic set to reduce MSE (skipped if mock pipeline).
 5. Iterative forecasting: use --iterative-step to autoregressively roll out horizon in smaller
        chunks (can reduce phase drift on longer horizons). Mutually exclusive with standard pred_len path.

Usage examples:
    Baseline single run:
        python chronosx_sine_cosine_eval.py --model-size large --seq-len 128 --pred-len 32
    Multi-period, normalization comparison & uncertainty:
        python chronosx_sine_cosine_eval.py --model-size large --seq-len 128 --pred-len 64 \
                --periods 3 --compare-norm --uncertainty --num-samples 50
    Calibration & iterative rollout:
        python chronosx_sine_cosine_eval.py --model-size large --seq-len 128 --pred-len 64 \
                --iterative-step 8 --calibrate-steps 200
"""
from __future__ import annotations

import argparse
import math
from typing import Dict, List, Tuple, Optional, Iterable

import torch
from argparse import Namespace
import statistics

from models.chronosx_autoformer import ChronosXAutoformer
from utils.normalization import TSNormalizer


def generate_signal(seq_len: int, pred_len: int, kind: str = "sine", freq: float = 1.0, phase: float = 0.0,
                    periods: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a univariate periodic signal and its future horizon.

    Parameters
    ----------
    seq_len : int
        Context length.
    pred_len : int
        Prediction horizon length.
    kind : str
        'sine' or 'cosine'.
    freq : float
        Frequency multiplier.
    phase : float
        Phase offset (radians).
    periods : float
        Number of base (2π) periods spanned by the context window (recommendation #1).
    """
    span_ctx = 2 * math.pi * periods
    t_ctx = torch.linspace(0, span_ctx, seq_len)
    # Future continues seamlessly after context end
    t_fut = torch.linspace(span_ctx, span_ctx + span_ctx * (pred_len / seq_len), pred_len)
    if kind == "sine":
        ctx = torch.sin(freq * t_ctx + phase)
        fut = torch.sin(freq * t_fut + phase)
    else:
        ctx = torch.cos(freq * t_ctx + phase)
        fut = torch.cos(freq * t_fut + phase)
    return ctx, fut


def build_model(seq_len: int, pred_len: int, model_size: str, uncertainty: bool, num_samples: int) -> ChronosXAutoformer:
    """Instantiate a ChronosXAutoformer for univariate forecasting."""
    cfg = Namespace(
        seq_len=seq_len,
        label_len=max(1, seq_len // 2),
        pred_len=pred_len,
        enc_in=1,
        dec_in=1,
        c_out=1,
        d_model=128,
        chronosx_model_size=model_size,
        uncertainty_enabled=uncertainty,
    num_samples=num_samples if uncertainty else 1,
        task_name="long_term_forecast",
        enable_multivariate_adapter=False,
        enable_covariate_residual=False,
        force_mock_pipeline=False,
        disable_hf_integration=True,
    )
    model = ChronosXAutoformer(cfg)
    model.eval()
    return model


def prepare_inputs(context: torch.Tensor, future_len: int, label_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare encoder/decoder tensors matching ChronosXAutoformer forward signature."""
    seq_len = context.size(0)
    pred_len = future_len
    label_len_clamped = min(label_len, seq_len)
    dec_hist = context[-label_len_clamped:]
    dec_future_pad = torch.zeros(pred_len)
    x_dec_full = torch.cat([dec_hist, dec_future_pad], dim=0)

    x_enc = context.view(1, seq_len, 1)
    x_mark_enc = torch.zeros(1, seq_len, 1)
    x_dec = x_dec_full.view(1, label_len_clamped + pred_len, 1)
    x_mark_dec = torch.zeros(1, label_len_clamped + pred_len, 1)
    return x_enc, x_mark_enc, x_dec, x_mark_dec


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute basic regression & directional metrics."""
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    eps = 1e-8
    sign_acc = torch.mean(((pred * target) > 0).float()).item()
    pred_c = pred - pred.mean()
    targ_c = target - target.mean()
    denom = torch.sqrt(torch.sum(pred_c ** 2) * torch.sum(targ_c ** 2) + eps)
    corr = (torch.sum(pred_c * targ_c) / denom).item() if denom > 0 else 0.0
    return {"mse": mse, "mae": mae, "sign_acc": sign_acc, "corr": corr}


def evaluate(model: ChronosXAutoformer, seq_len: int, pred_len: int, normalizer: Optional[TSNormalizer] = None,
             periods: float = 1.0, iterative_step: int = 0) -> List[Dict[str, float]]:
    """Run evaluation across several (kind,freq,phase) combinations.

    If a normalizer is provided and fitted, both contextual and future windows
    are transformed before feeding to the model, and predictions are inverse
    transformed prior to metric computation. This keeps reported metrics in
    original (unnormalized) scale while allowing the model to consume
    normalized inputs.
    """
    configs = [
        ("sine", 1.0, 0.0),
        ("sine", 2.0, 0.3),
        ("cosine", 1.0, 0.0),
        ("cosine", 0.5, 1.0),
    ]
    metrics_list: List[Dict[str, float]] = []
    for kind, freq, phase in configs:
        ctx, fut = generate_signal(seq_len, pred_len, kind=kind, freq=freq, phase=phase, periods=periods)

        # Optional normalization (fit only once on first full context+future concatenation)
        if normalizer and not normalizer.fitted():
            # Fit only on context (avoid peeking into future)
            import pandas as pd  # local import to avoid global dependency if unused
            ctx_df_fit = pd.DataFrame(ctx.unsqueeze(1).numpy(), columns=["value"])
            normalizer.fit(ctx_df_fit)
        if normalizer and normalizer.fitted():
            import pandas as pd
            ctx_df = pd.DataFrame(ctx.unsqueeze(1).numpy(), columns=["value"])
            fut_df = pd.DataFrame(fut.unsqueeze(1).numpy(), columns=["value"])
            ctx_norm = torch.from_numpy(normalizer.transform(ctx_df).values.squeeze())
            fut_orig = fut  # keep original for metric computation
            fut = fut_orig  # alias clarity
            ctx = ctx_norm
        if iterative_step and iterative_step < pred_len:
            pred = iterative_rollout(model, ctx, pred_len, iterative_step, model.configs.label_len)
        else:
            x_enc, x_mark_enc, x_dec, x_mark_dec = prepare_inputs(ctx, pred_len, model.configs.label_len)
            with torch.no_grad():
                out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            pred = out.squeeze(0).squeeze(-1)
        # Inverse transform prediction if normalization was applied
        if normalizer and normalizer.fitted():
            import pandas as pd
            pred_df = pd.DataFrame(pred.unsqueeze(1).numpy(), columns=["value"])
            pred_inv = torch.from_numpy(normalizer.inverse_transform(pred_df).values.squeeze())
            pred = pred_inv
        metrics = compute_metrics(pred, fut)
        metrics.update({"kind": kind, "freq": freq, "phase": phase})
        metrics_list.append(metrics)
    return metrics_list


def main() -> None:
    parser = argparse.ArgumentParser(description="ChronosXAutoformer sine/cosine evaluation")
    parser.add_argument('--model-size', type=str, default='large', choices=['tiny', 'mini', 'small', 'base', 'large'])
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--pred-len', type=int, default=32)
    parser.add_argument('--uncertainty', action='store_true', help='Enable uncertainty sampling (recommendation #3)')
    parser.add_argument('--num-samples', type=int, default=30, help='Number of samples when uncertainty enabled (recommendation #3)')
    parser.add_argument('--force-mock', action='store_true', help='Force use of mock Chronos pipeline even if chronos is installed')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--normalize', action='store_true', help='Apply TSNormalizer (standard) to synthetic data before modeling (metrics reported on original scale).')
    parser.add_argument('--norm-mode', type=str, default='standard', choices=['standard','minmax'], help='Normalization mode for synthetic evaluation.')
    parser.add_argument('--compare-norm', action='store_true', help='Compare none,standard,minmax modes (recommendation #2)')
    parser.add_argument('--periods', type=float, default=1.0, help='Number of full base periods in context (recommendation #1)')
    parser.add_argument('--calibrate-steps', type=int, default=0, help='Train lightweight residual calibrator steps (recommendation #4)')
    parser.add_argument('--iterative-step', type=int, default=0, help='Iterative rollout step size (<pred_len) for autoregressive forecasting (recommendation #5)')
    args = parser.parse_args()

    if args.force_mock:
        # Induce mock by setting env flag in config namespace after init
        model = build_model(args.seq_len, args.pred_len, args.model_size, args.uncertainty, args.num_samples)
        model.force_mock_pipeline = True
        model._create_mock_pipeline()
    else:
        model = build_model(args.seq_len, args.pred_len, args.model_size, args.uncertainty, args.num_samples)
    if getattr(model, 'is_mock', True):
        info = model.get_chronos_info()
        print("[WARN] Using mock ChronosX pipeline — forecasts are random baselines.")
        print(f"       chronos_available={info['chronos_available']} init_error={info.get('init_error')}")
        print("       To enable real model: pip install -U chronos-forecasting (ensure PyTorch version & Python compatible)")

    def run_single(norm_mode: Optional[str]) -> Tuple[str, List[Dict[str, float]]]:
        normalizer_local: Optional[TSNormalizer] = None
        if norm_mode is not None:
            try:
                normalizer_local = TSNormalizer(norm_mode)
            except Exception as e:
                print(f"[WARN] Failed to init normalizer {norm_mode}: {e}")
                normalizer_local = None
        metrics = evaluate(model, args.seq_len, args.pred_len, normalizer_local, periods=args.periods,
                            iterative_step=args.iterative_step)
        return norm_mode or 'none', metrics

    norm_sets: List[Tuple[str, List[Dict[str, float]]]] = []
    if args.compare-norm:
        for nm in [None, 'standard', 'minmax']:
            label, mets = run_single(nm)
            norm_sets.append((label, mets))
    else:
        label, mets = run_single(args.norm_mode if args.normalize else None)
        norm_sets.append((label, mets))

    # Optional lightweight calibration (recommendation #4)
    if args.calibrate_steps > 0 and not getattr(model, 'is_mock', True):
        calibrate_model(model, norm_sets[0][1], steps=args.calibrate_steps)
        # Re-run after calibration for primary setting only
        label, mets = run_single(args.norm_mode if args.normalize else None)
        norm_sets[0] = (label + '+cal', mets)

    print(f"\n=== ChronosXAutoformer ({args.model_size}) Sine/Cosine Evaluation ===")
    for label, metrics_list in norm_sets:
        print(f"\n-- Normalization: {label} --")
        for m in metrics_list:
            print(f"{m['kind']:6s} f={m['freq']:<4} phase={m['phase']:<4}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  Corr={m['corr']:.3f}  SignAcc={m['sign_acc']:.3f}")
        avg_mse = sum(m['mse'] for m in metrics_list) / len(metrics_list)
        avg_corr = sum(m['corr'] for m in metrics_list) / len(metrics_list)
        print(f"Average MSE: {avg_mse:.4f}  Average Corr: {avg_corr:.3f}")
    if args.iterative_step:
        print(f"(Iterative rollout step size={args.iterative_step})")
    if args.periods != 1.0:
        print(f"(Context periods={args.periods})")

    if args.uncertainty and hasattr(model, 'last_uncertainty_results'):
        uq = model.last_uncertainty_results
        if isinstance(uq, dict) and 'std' in uq:
            print("Uncertainty sample (std tensor shape):", uq['std'].shape)


# ---------------------------------------------------------------------------
# Iterative forecasting (recommendation #5)
# ---------------------------------------------------------------------------
def iterative_rollout(model: ChronosXAutoformer, context: torch.Tensor, total_pred_len: int,
                      step: int, label_len: int) -> torch.Tensor:
    """Autoregressively roll out predictions in smaller steps.

    Parameters
    ----------
    model : ChronosXAutoformer
    context : (seq_len,) tensor
    total_pred_len : int
    step : int
        Step size (< total_pred_len) for each iterative forecasting call.
    label_len : int
        Label/history length for decoder.
    """
    preds: List[torch.Tensor] = []
    cur_ctx = context.clone()
    produced = 0
    while produced < total_pred_len:
        cur_step = min(step, total_pred_len - produced)
        x_enc, x_mark_enc, x_dec, x_mark_dec = prepare_inputs(cur_ctx, cur_step, label_len)
        with torch.no_grad():
            out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        step_pred = out.squeeze(0).squeeze(-1)[:cur_step]
        preds.append(step_pred)
        produced += cur_step
        # Append new predictions to context (sliding window)
        cur_ctx = torch.cat([cur_ctx, step_pred])
        if cur_ctx.size(0) > context.size(0):
            cur_ctx = cur_ctx[-context.size(0):]
    return torch.cat(preds, dim=0)


# ---------------------------------------------------------------------------
# Lightweight calibration (recommendation #4)
# ---------------------------------------------------------------------------
def calibrate_model(model: ChronosXAutoformer, metrics_source: List[Dict[str, float]], steps: int = 100) -> None:
    """Train a simple per-horizon affine residual (scale & bias) to reduce MSE.

    For simplicity we perform a few gradient steps on synthetic samples using the
    current model as a fixed feature generator. Calibration parameters are stored
    on model as calibration_scale & calibration_bias and applied in postprocess.
    (Non-invasive: only affects this evaluation session.)
    """
    if getattr(model, 'is_mock', True):
        print("[INFO] Skipping calibration: mock pipeline.")
        return
    if hasattr(model, 'calibration_scale'):
        return  # already calibrated
    pred_len = model.pred_len
    # Initialize parameters
    model.calibration_scale = torch.nn.Parameter(torch.ones(pred_len))  # type: ignore
    model.calibration_bias = torch.nn.Parameter(torch.zeros(pred_len))  # type: ignore
    optimizer = torch.optim.Adam([model.calibration_scale, model.calibration_bias], lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    for step_idx in range(steps):
        # Generate a random sine config for calibration batch
        ctx, fut = generate_signal(model.seq_len, pred_len, kind='sine', freq=1.0 + 0.5*torch.rand(1).item(),
                                   phase=torch.rand(1).item()*math.pi, periods=1.0)
        x_enc, x_mark_enc, x_dec, x_mark_dec = prepare_inputs(ctx, pred_len, model.configs.label_len)
        with torch.no_grad():
            out = model(x_enc, x_mark_enc, x_dec, x_mark_dec).squeeze(0).squeeze(-1)
        adjusted = out * model.calibration_scale + model.calibration_bias
        loss = loss_fn(adjusted, fut)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step_idx + 1) % max(1, steps // 5) == 0:
            print(f"[Calibrate] step {step_idx+1}/{steps} loss={loss.item():.4f}")
    # Monkey patch postprocess to apply calibration
    original_post = model.postprocess_prediction
    def _patched(pred: torch.Tensor) -> torch.Tensor:
        pred_adj = pred.clone()
        # pred shape [B, pred_len, c_out]
        if pred_adj.size(1) == pred_len and pred_adj.size(-1) >= 1:
            scale = model.calibration_scale.to(pred_adj.device)
            bias = model.calibration_bias.to(pred_adj.device)
            pred_adj[..., 0] = pred_adj[..., 0] * scale + bias
        return original_post(pred_adj)
    model.postprocess_prediction = _patched  # type: ignore
    print("[Calibrate] Calibration applied.")


if __name__ == '__main__':
    main()