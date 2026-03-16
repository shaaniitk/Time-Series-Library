"""
TFT End-to-End Ablation Study on Synthetic Multi-Scale Wave Data
================================================================

Three waves with varying dynamics:
  - Low frequency  (period ≈ 96): slow-moving baseline oscillation
  - Medium frequency (period ≈ 24): diurnal-scale cycle
  - High frequency  (period ≈ 6):  fast burst-like oscillation

Each wave is characterized by 4 time-varying parameters:
  1. sin(phase(t))       — wave position (sine component)
  2. cos(phase(t))       — wave position (cosine component)
  3. ω(t) = d(phase)/dt  — instantaneous angular velocity
  4. A(t)                — time-varying amplitude

Phase speeds drift slowly (±15%) around their default values, and
amplitudes are modulated by low-frequency envelopes.

Two complex target variables:
  y1 — "Phase Coherence": depends on relative phase positions,
       amplitude ratios, regime gating, and cross-frequency products
  y2 — "Velocity Momentum": depends on angular velocity ratios,
       threshold transitions, and non-linear cross-frequency mixing

Both targets include irreducible stochastic noise (σ = 0.15).

We test multiple model configurations to identify which architectural
features help capture these complex multi-scale dynamics.
"""

import math
import sys
import time
import unittest
from collections import OrderedDict
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import TemporalFusionTransformer as tsl_tft
from utils.losses import QuantileLoss

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

SEQ_LEN = 96
LABEL_LEN = 48
PRED_LEN = 24
ENC_IN = 16       # 2 targets + 12 wave params + 2 interaction features
C_OUT = 2         # 2 complex target variables
D_MODEL = 128
N_HEADS = 8
KNOWN_LEN = 8     # 6 deterministic periodic + trend + modulation
N_TRAIN = 1024
N_VAL = 256
BATCH_SIZE = 64
EPOCHS = 30
LR = 5e-4
NOISE_STD = 0.15
SEED = 42

# Wave periods
PERIOD_LOW = 96.0
PERIOD_MED = 24.0
PERIOD_HIGH = 6.0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Synthetic Wave Data Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_e2e_data(
    n_samples,
    seq_len=SEQ_LEN,
    label_len=LABEL_LEN,
    pred_len=PRED_LEN,
    noise_std=NOISE_STD,
    seed=SEED,
):
    """
    Generate synthetic 3-wave data with 2 complex target variables.

    Returns
    -------
    dict with keys: x_enc, x_mark_enc, x_dec, x_mark_dec, y_future
        Each is a tensor ready for TFT consumption.
    """
    torch.manual_seed(seed)
    N = n_samples
    T = seq_len + pred_len
    t_idx = torch.arange(T, dtype=torch.float32).view(1, T, 1)  # [1, T, 1]

    # ── Per-sample random wave parameters ──
    # Phase offsets (random starting position per sample)
    phi_low = 2 * math.pi * torch.rand(N, 1, 1)
    phi_med = 2 * math.pi * torch.rand(N, 1, 1)
    phi_high = 2 * math.pi * torch.rand(N, 1, 1)

    # Base angular velocities
    omega_base_low = 2 * math.pi / PERIOD_LOW
    omega_base_med = 2 * math.pi / PERIOD_MED
    omega_base_high = 2 * math.pi / PERIOD_HIGH

    # Per-sample velocity scales (±15-20% variation around base)
    vel_scale_low = 0.85 + 0.30 * torch.rand(N, 1, 1)
    vel_scale_med = 0.85 + 0.30 * torch.rand(N, 1, 1)
    vel_scale_high = 0.80 + 0.40 * torch.rand(N, 1, 1)

    # Per-sample amplitude scales
    A_low_base = 1.0 + 0.3 * torch.rand(N, 1, 1)
    A_med_base = 0.6 + 0.3 * torch.rand(N, 1, 1)
    A_high_base = 0.25 + 0.25 * torch.rand(N, 1, 1)

    # ── Time-varying angular velocity with slow drift ──
    modulation_T = float(T)
    drift_low = 1.0 + 0.06 * torch.sin(
        2 * math.pi * t_idx / modulation_T + 0.3 * phi_low
    )
    drift_med = 1.0 + 0.10 * torch.sin(
        2 * math.pi * t_idx / modulation_T + 0.5 * phi_med
    )
    drift_high = 1.0 + 0.15 * torch.sin(
        2 * math.pi * t_idx / (modulation_T * 0.7) + 0.7 * phi_high
    )

    # Instantaneous angular velocities [N, T, 1]
    omega_low = omega_base_low * vel_scale_low * drift_low
    omega_med = omega_base_med * vel_scale_med * drift_med
    omega_high = omega_base_high * vel_scale_high * drift_high

    # Cumulative phase
    phase_low = phi_low + torch.cumsum(omega_low, dim=1)
    phase_med = phi_med + torch.cumsum(omega_med, dim=1)
    phase_high = phi_high + torch.cumsum(omega_high, dim=1)

    # ── Time-varying amplitudes (slow envelope modulation) ──
    A_low = A_low_base * (
        1.0 + 0.10 * torch.sin(2 * math.pi * t_idx / (PERIOD_LOW * 2) + phi_low)
    )
    A_med = A_med_base * (
        1.0 + 0.15 * torch.sin(2 * math.pi * t_idx / (PERIOD_MED * 3) + phi_med)
    )
    A_high = A_high_base * (
        1.0 + 0.20 * torch.sin(2 * math.pi * t_idx / (PERIOD_HIGH * 5) + phi_high)
    )

    # ── Wave components ──
    low_sin = A_low * torch.sin(phase_low)
    low_cos = A_low * torch.cos(phase_low)
    med_sin = A_med * torch.sin(phase_med)
    med_cos = A_med * torch.cos(phase_med)
    high_sin = A_high * torch.sin(phase_high)
    high_cos = A_high * torch.cos(phase_high)

    # ── Phase differences (wrapped to [-π, π]) ──
    def wrap(x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    delta_lm = wrap(phase_low - phase_med)
    delta_mh = wrap(phase_med - phase_high)
    delta_lh = wrap(phase_low - phase_high)

    # ── Velocity and amplitude ratios ──
    eps = 1e-6
    ratio_ml = omega_med / (omega_low.abs() + eps)
    ratio_hm = omega_high / (omega_med.abs() + eps)
    ratio_hl = omega_high / (omega_low.abs() + eps)
    amp_ratio_ml = A_med / (A_low + eps)
    amp_ratio_hl = A_high / (A_low + eps)
    amp_ratio_hm = A_high / (A_med + eps)

    # ── Regime gating (soft state based on phase alignment) ──
    coherence_lm = torch.cos(delta_lm)
    regime = torch.sigmoid(2.0 * (coherence_lm + 0.5 * torch.sin(delta_mh)))
    burst = torch.sigmoid(
        3.0 * (torch.cos(delta_lh) + 0.3 * torch.tanh(ratio_hl - 3.0))
    )

    # ══════════════════════════════════════════════
    # Target 1: Phase-Coherence Variable
    # ══════════════════════════════════════════════
    # Driven by relative positions of waves, amplitude ratios, regime gating
    y1 = (
        0.30 * torch.sin(delta_lm) * torch.cos(delta_mh)
        + 0.20 * amp_ratio_ml * torch.cos(delta_lm)
        + 0.15 * regime * (low_sin + 0.5 * med_sin)
        + 0.12 * burst * high_sin
        + 0.10 * low_sin * med_cos
        + 0.08 * torch.tanh(ratio_ml - 2.5) * med_sin
        + 0.05 * amp_ratio_hl * torch.sin(delta_lh)
    )

    # ══════════════════════════════════════════════
    # Target 2: Velocity-Momentum Variable
    # ══════════════════════════════════════════════
    # Driven by speed ratios, threshold effects, cross-frequency mixing
    y2 = (
        0.25 * torch.tanh(ratio_hm - 2.0)
        + 0.20
        * torch.sigmoid(3.0 * (ratio_ml - 1.5))
        * torch.sin(delta_lh)
        + 0.15 * amp_ratio_hm * amp_ratio_ml * torch.cos(delta_mh)
        + 0.12 * low_cos * high_sin
        + 0.10 * torch.tanh(omega_high - omega_low)
        + 0.08 * burst * (ratio_hl - ratio_ml)
        + 0.05 * regime * med_cos * high_cos
        + 0.05 * torch.sin(phase_low * 0.5 + phase_high * 0.3)
    )

    # Add stochastic noise
    y1 = y1 + noise_std * torch.randn_like(y1)
    y2 = y2 + noise_std * torch.randn_like(y2)
    targets = torch.cat([y1, y2], dim=-1)  # [N, T, 2]

    # ── Normalized feature variants for model input ──
    # Angular velocity normalized to ~1.0 center
    omega_low_feat = omega_low / omega_base_low
    omega_med_feat = omega_med / omega_base_med
    omega_high_feat = omega_high / omega_base_high
    # Phase difference and velocity ratio (already reasonable scale)
    delta_lm_feat = delta_lm / math.pi       # normalize to [-1, 1]
    ratio_hm_feat = ratio_hm / (PERIOD_MED / PERIOD_HIGH)  # normalize to ~1.0

    # ── Encoder features (16 channels) ──
    # Layout: [y1, y2, low_sin, low_cos, med_sin, med_cos, high_sin, high_cos,
    #          ω_low, ω_med, ω_high, A_low, A_med, A_high, Δφ_lm, ratio_hm]
    history_targets = targets[:, :seq_len, :]  # [N, seq_len, 2]
    history_features = torch.cat([
        low_sin[:, :seq_len],
        low_cos[:, :seq_len],
        med_sin[:, :seq_len],
        med_cos[:, :seq_len],
        high_sin[:, :seq_len],
        high_cos[:, :seq_len],
        omega_low_feat[:, :seq_len],
        omega_med_feat[:, :seq_len],
        omega_high_feat[:, :seq_len],
        A_low[:, :seq_len],
        A_med[:, :seq_len],
        A_high[:, :seq_len],
        delta_lm_feat[:, :seq_len],
        ratio_hm_feat[:, :seq_len],
    ], dim=-1)  # [N, seq_len, 14]
    x_enc = torch.cat([history_targets, history_features], dim=-1)  # [N, seq_len, 16]

    # ── Known covariates (8 channels, deterministic → available past + future) ──
    # Deterministic periodic features at BASE frequencies (no random perturbation)
    full_time = t_idx.expand(N, -1, -1)  # [N, T, 1]
    known_features = torch.cat([
        torch.sin(omega_base_low * full_time),
        torch.cos(omega_base_low * full_time),
        torch.sin(omega_base_med * full_time),
        torch.cos(omega_base_med * full_time),
        torch.sin(omega_base_high * full_time),
        torch.cos(omega_base_high * full_time),
        full_time / modulation_T,                                    # linear trend
        torch.sin(2 * math.pi * full_time / modulation_T),          # full-period cycle
    ], dim=-1)  # [N, T, 8]

    x_mark_enc = known_features[:, :seq_len, :]           # [N, seq_len, 8]
    x_mark_dec = known_features[:, seq_len - label_len:seq_len + pred_len, :]  # [N, label+pred, 8]

    # ── Decoder input ──
    label_context = targets[:, seq_len - label_len:seq_len, :]  # [N, label_len, 2]
    x_dec = torch.cat([
        label_context,
        torch.zeros(N, pred_len, C_OUT),
    ], dim=1)  # [N, label_len+pred_len, 2]

    # ── Ground truth ──
    y_future = targets[:, seq_len:seq_len + pred_len, :]  # [N, pred_len, 2]

    return {
        "x_enc": x_enc,
        "x_mark_enc": x_mark_enc,
        "x_dec": x_dec,
        "x_mark_dec": x_mark_dec,
        "y_future": y_future,
    }


def make_dataloaders(batch_size=BATCH_SIZE):
    """Generate train and validation DataLoaders."""
    train_data = generate_e2e_data(N_TRAIN, seed=SEED)
    val_data = generate_e2e_data(N_VAL, seed=SEED + 1000)
    train_ds = TensorDataset(
        train_data["x_enc"],
        train_data["x_mark_enc"],
        train_data["x_dec"],
        train_data["x_mark_dec"],
        train_data["y_future"],
    )
    val_ds = TensorDataset(
        val_data["x_enc"],
        val_data["x_mark_enc"],
        val_data["x_dec"],
        val_data["x_mark_dec"],
        val_data["y_future"],
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Model Configuration Factory
# ═══════════════════════════════════════════════════════════════════════════════

def e2e_base_config():
    """Minimal TFT configuration — LSTM backbone, no feature flags enabled."""
    return SimpleNamespace(
        task_name="long_term_forecast",
        data="synthetic_e2e",
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        enc_in=ENC_IN,
        dec_in=C_OUT,
        c_out=C_OUT,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dropout=0.1,
        embed="timeF",
        freq="h",
        e_layers=3,
        # All features OFF for baseline
        tft_use_swiglu=False,
        tft_full_attention=False,
        tft_dual_attention_fusion=False,
        tft_use_explicit_cross_attention=False,
        tft_cross_attention_type="full",
        tft_attention_position_bias="none",
        tft_attention_backend="exact",
        tft_rope_base=10000.0,
        tft_alibi_scale=1.0,
        tft_use_revin=False,
        tft_revin_affine=True,
        tft_use_quantile_head=False,
        tft_output_quantiles=[0.1, 0.5, 0.9],
        tft_use_lag_attention=False,
        tft_lag_scales=[1, 2, 4],
        tft_temporal_backbone="lstm",
        tft_temporal_backbone_layers=3,
        tft_temporal_kernel_size=3,
        tft_temporal_hidden_size=0,
        tft_use_higher_order=False,
        tft_interaction_order=2,
        tft_interaction_rank=32,
        tft_use_regime_moe=False,
        tft_num_regimes=3,
        tft_num_moe_experts=4,
        tft_moe_top_k=2,
        tft_moe_hidden_size=0,
        tft_moe_noise_epsilon=1e-2,
        tft_moe_aux_loss_coeff=0.0,
        tft_payload_stack_layers=False,
        tft_cross_variable_mixing=False,
        tft_vsn_residual_bypass=True,
        tft_vsn_n_selection_heads=1,
        tft_allow_custom_known=True,
        tft_known_len=KNOWN_LEN,
        tft_known_max_channels=64,
        tft_observed_pos=list(range(ENC_IN)),
        tft_static_pos=[],
        tft_target_pos=[0, 1],
        tft_use_fft_branch=False,
        tft_fft_modes=32,
        tft_fft_mode_select="low",
        tft_stochastic_depth_rate=0.0,
        tft_gradient_checkpointing=False,
        tft_use_temporal_compression=False,
        tft_tc_stride=2,
        tft_tc_threshold=256,
        tft_mlp_quantile_projection=False,
        tft_quantile_projection_ff_size=0,
    )


def make_config(overrides):
    """Create a config with specific overrides on top of the base."""
    cfg = e2e_base_config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ── Ablation configurations ──
ABLATIONS = OrderedDict([
    # Individual Feature Tests (one feature at a time vs baseline)
    ("baseline", {}),
    ("revin", {
        "tft_use_revin": True,
    }),
    ("fft_branch", {
        "tft_use_fft_branch": True,
        "tft_fft_modes": 16,
    }),
    ("higher_order", {
        "tft_use_higher_order": True,
        "tft_interaction_order": 2,
    }),
    ("regime_moe", {
        "tft_use_regime_moe": True,
        "tft_moe_aux_loss_coeff": 0.01,
    }),
    ("multihead_vsn", {
        "tft_vsn_n_selection_heads": 2,
    }),
    ("dual_attention", {
        "tft_full_attention": True,
        "tft_dual_attention_fusion": True,
    }),
    ("cross_attention", {
        "tft_use_explicit_cross_attention": True,
    }),
    ("gated_tcn", {
        "tft_temporal_backbone": "gated_tcn",
    }),
    ("hybrid_backbone", {
        "tft_temporal_backbone": "hybrid_tcn_lstm",
    }),
    # Combination Configs
    ("combo_spectral", {
        "tft_use_revin": True,
        "tft_use_fft_branch": True,
        "tft_fft_modes": 16,
        "tft_use_higher_order": True,
    }),
    ("combo_attention", {
        "tft_use_revin": True,
        "tft_full_attention": True,
        "tft_dual_attention_fusion": True,
        "tft_use_lag_attention": True,
        "tft_use_explicit_cross_attention": True,
    }),
    # Full Stack
    ("full_stack", {
        "tft_use_revin": True,
        "tft_use_swiglu": True,
        "tft_full_attention": True,
        "tft_dual_attention_fusion": True,
        "tft_use_explicit_cross_attention": True,
        "tft_use_lag_attention": True,
        "tft_use_fft_branch": True,
        "tft_fft_modes": 16,
        "tft_use_higher_order": True,
        "tft_use_regime_moe": True,
        "tft_moe_aux_loss_coeff": 0.01,
        "tft_vsn_n_selection_heads": 2,
        "tft_cross_variable_mixing": True,
        "tft_temporal_backbone": "hybrid_tcn_lstm",
    }),
    # Full + Quantile Head
    ("full_quantile", {
        "tft_use_revin": True,
        "tft_use_swiglu": True,
        "tft_full_attention": True,
        "tft_dual_attention_fusion": True,
        "tft_use_explicit_cross_attention": True,
        "tft_use_lag_attention": True,
        "tft_use_fft_branch": True,
        "tft_fft_modes": 16,
        "tft_use_higher_order": True,
        "tft_use_regime_moe": True,
        "tft_moe_aux_loss_coeff": 0.01,
        "tft_vsn_n_selection_heads": 2,
        "tft_cross_variable_mixing": True,
        "tft_temporal_backbone": "hybrid_tcn_lstm",
        "tft_use_quantile_head": True,
        "tft_mlp_quantile_projection": True,
        "tft_quantile_projection_ff_size": D_MODEL,
    }),
])


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_evaluate(name, cfg, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    """
    Train a TFT config and return comprehensive metrics.

    Returns dict with: train_losses, val_losses, train_mse_final, val_mse_final,
    improvement_pct, generalization_gap, stable, elapsed_s, n_params,
    quantile_val_loss (if applicable), model (trained model ref).
    """
    torch.manual_seed(SEED)
    model = tsl_tft.Model(cfg)
    n_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_fn = nn.MSELoss()
    quantile_fn = None
    if cfg.tft_use_quantile_head:
        quantile_fn = QuantileLoss(cfg.tft_output_quantiles)

    train_losses = []
    val_losses = []
    quantile_val_losses = []
    stable = True
    t0 = time.time()

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_enc, x_mark_enc, x_dec, x_mark_dec, y in train_loader:
            optimizer.zero_grad()
            out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            pred = out[:, -cfg.pred_len:, :]
            loss = mse_fn(pred, y)
            # Add quantile loss if quantile head is active
            if quantile_fn is not None and model.last_quantile_predictions is not None:
                q_pred = model.last_quantile_predictions
                loss = loss + 0.5 * quantile_fn(q_pred, y)
            # Add MoE auxiliary loss
            aux = getattr(model, "last_moe_aux_loss", None)
            if torch.is_tensor(aux):
                loss = loss + cfg.tft_moe_aux_loss_coeff * aux

            if not torch.isfinite(loss):
                stable = False
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if not stable:
            break
        train_losses.append(epoch_loss / max(n_batches, 1))

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        q_val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_enc, x_mark_enc, x_dec, x_mark_dec, y in val_loader:
                out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                pred = out[:, -cfg.pred_len:, :]
                val_loss += mse_fn(pred, y).item()
                if quantile_fn is not None and model.last_quantile_predictions is not None:
                    q_val_loss += quantile_fn(model.last_quantile_predictions, y).item()
                n_val += 1
        val_losses.append(val_loss / max(n_val, 1))
        if quantile_fn is not None:
            quantile_val_losses.append(q_val_loss / max(n_val, 1))

    elapsed = time.time() - t0

    # Compute summary metrics
    if stable and len(train_losses) >= 2:
        improvement = (train_losses[0] - train_losses[-1]) / (train_losses[0] + 1e-12) * 100
        gen_gap = (val_losses[-1] - train_losses[-1]) / (train_losses[-1] + 1e-12) * 100
    else:
        improvement = 0.0
        gen_gap = float("inf")

    return {
        "name": name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "quantile_val_losses": quantile_val_losses,
        "train_mse_0": train_losses[0] if train_losses else float("nan"),
        "train_mse_final": train_losses[-1] if train_losses else float("nan"),
        "val_mse_final": val_losses[-1] if val_losses else float("nan"),
        "improvement_pct": improvement,
        "generalization_gap_pct": gen_gap,
        "stable": stable,
        "elapsed_s": elapsed,
        "n_params": n_params,
        "model": model,
        "cfg": cfg,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Interpretation Analysis
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "y1_hist", "y2_hist",
    "low_sin", "low_cos", "med_sin", "med_cos", "high_sin", "high_cos",
    "ω_low", "ω_med", "ω_high",
    "A_low", "A_med", "A_high",
    "Δφ_lm", "ratio_hm",
]

KNOWN_NAMES = [
    "sin(ω₀_low·t)", "cos(ω₀_low·t)",
    "sin(ω₀_med·t)", "cos(ω₀_med·t)",
    "sin(ω₀_high·t)", "cos(ω₀_high·t)",
    "trend", "modulation",
]


def run_interpretation(model, sample_loader):
    """Run a forward pass with interpretation and extract key diagnostics."""
    model.eval()
    x_enc, x_mark_enc, x_dec, x_mark_dec, y = next(iter(sample_loader))
    with torch.no_grad():
        payload = model(
            x_enc[:4], x_mark_enc[:4], x_dec[:4], x_mark_dec[:4],
            return_interpretation=True,
        )
    return payload


def print_interpretation(payload, config_name):
    """Print interpretation diagnostics for a given config."""
    print(f"\n  Interpretation Analysis for '{config_name}':")

    # VSN variable importance
    hist_w = payload.get("history_vsn_weights")
    if hist_w is not None:
        if hist_w.ndim == 4:
            # Multi-head: [B, T, K, C] → average over B, T, K
            avg_w = hist_w.mean(dim=(0, 1, 2))
        else:
            # Single-head: [B, T, C] → average over B, T
            avg_w = hist_w.mean(dim=(0, 1))
        n_vars = min(len(avg_w), len(FEATURE_NAMES) + len(KNOWN_NAMES))
        var_names = FEATURE_NAMES + KNOWN_NAMES
        ranked = sorted(range(n_vars), key=lambda i: avg_w[i].item(), reverse=True)
        print("    History VSN importance (top 8):")
        for rank, idx in enumerate(ranked[:8]):
            name = var_names[idx] if idx < len(var_names) else f"var_{idx}"
            print(f"      {rank+1}. {name:18s} {avg_w[idx]:.4f}")

    future_w = payload.get("future_vsn_weights")
    if future_w is not None:
        if future_w.ndim == 4:
            avg_fw = future_w.mean(dim=(0, 1, 2))
        else:
            avg_fw = future_w.mean(dim=(0, 1))
        n_fvars = min(len(avg_fw), len(KNOWN_NAMES))
        ranked_f = sorted(range(n_fvars), key=lambda i: avg_fw[i].item(), reverse=True)
        print("    Future VSN importance (top 5):")
        for rank, idx in enumerate(ranked_f[:5]):
            name = KNOWN_NAMES[idx] if idx < len(KNOWN_NAMES) else f"known_{idx}"
            print(f"      {rank+1}. {name:18s} {avg_fw[idx]:.4f}")

    # Attention fusion alpha (if dual attention enabled)
    alpha = payload.get("attention_fusion_alpha")
    if alpha is not None:
        print(f"    Attention fusion α (full vs interp): {alpha.mean().item():.4f}")

    # FFT gate
    fft_gate = payload.get("fft_gate_mean")
    if fft_gate is not None:
        print(f"    FFT gate mean: {fft_gate:.4f}")

    # Regime probabilities
    regime_probs = payload.get("regime_probabilities_pooled")
    if regime_probs is not None:
        avg_regime = regime_probs.mean(dim=0)
        regime_str = ", ".join(f"{p:.3f}" for p in avg_regime.tolist())
        print(f"    Regime probabilities: [{regime_str}]")

    # Expert routing
    expert_routing = payload.get("expert_routing")
    if expert_routing is not None:
        avg_routing = expert_routing.float().mean(dim=(0, 1))
        routing_str = ", ".join(f"{p:.3f}" for p in avg_routing.tolist())
        print(f"    Expert routing (avg): [{routing_str}]")

    # Quantile spread
    qp = payload.get("quantile_predictions")
    if qp is not None:
        spread = (qp[:, :, -1, :] - qp[:, :, 0, :]).mean().item()
        print(f"    Quantile spread (q90-q10 avg): {spread:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Results Summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results):
    """Print a comparative summary table of all ablation results."""
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 100)
    print(
        f"{'Config':<22s} │ {'Params':>7s} │ {'Train₀':>8s} │ {'Train_f':>8s} │ "
        f"{'Val_f':>8s} │ {'Improv%':>7s} │ {'GenGap%':>7s} │ {'Time':>5s} │ {'Status':>6s}"
    )
    print("─" * 22 + "─┼─" + "─" * 7 + "─┼─" + "─" * 8 + "─┼─" + "─" * 8 + "─┼─" +
          "─" * 8 + "─┼─" + "─" * 7 + "─┼─" + "─" * 7 + "─┼─" + "─" * 5 + "─┼─" + "─" * 6)

    for name, r in results.items():
        status = "OK" if r["stable"] else "FAIL"
        params_k = r["n_params"] / 1000
        print(
            f"{name:<22s} │ {params_k:>6.1f}k │ {r['train_mse_0']:>8.5f} │ "
            f"{r['train_mse_final']:>8.5f} │ {r['val_mse_final']:>8.5f} │ "
            f"{r['improvement_pct']:>6.1f}% │ {r['generalization_gap_pct']:>6.1f}% │ "
            f"{r['elapsed_s']:>4.1f}s │ {status:>6s}"
        )

    # Ranking by validation loss
    ranked = sorted(
        [(name, r) for name, r in results.items() if r["stable"]],
        key=lambda x: x[1]["val_mse_final"],
    )
    print("\n" + "─" * 60)
    print("RANKING BY VALIDATION MSE (lower is better):")
    print("─" * 60)
    baseline_val = results.get("baseline", {}).get("val_mse_final", float("inf"))
    for rank, (name, r) in enumerate(ranked):
        delta = (r["val_mse_final"] - baseline_val) / (baseline_val + 1e-12) * 100
        sign = "+" if delta > 0 else ""
        marker = " ◀ BEST" if rank == 0 else ""
        print(f"  {rank+1:2d}. {name:<22s}  val={r['val_mse_final']:.5f}  "
              f"({sign}{delta:.1f}% vs baseline){marker}")

    # Feature impact analysis
    print("\n" + "─" * 60)
    print("INDIVIDUAL FEATURE IMPACT (Δ val MSE vs baseline):")
    print("─" * 60)
    individual_features = [
        "revin", "fft_branch", "higher_order", "regime_moe",
        "multihead_vsn", "dual_attention", "cross_attention",
        "gated_tcn", "hybrid_backbone",
    ]
    impacts = []
    for feat in individual_features:
        if feat in results and results[feat]["stable"]:
            delta = results[feat]["val_mse_final"] - baseline_val
            pct = delta / (baseline_val + 1e-12) * 100
            impacts.append((feat, delta, pct))
    impacts.sort(key=lambda x: x[1])
    for feat, delta, pct in impacts:
        arrow = "↓" if delta < 0 else "↑"
        color_sign = "" if delta < 0 else "+"
        print(f"    {feat:<22s}  {arrow} {color_sign}{pct:.1f}%  ({color_sign}{delta:.5f})")

    # Failed configs
    failed = [name for name, r in results.items() if not r["stable"]]
    if failed:
        print(f"\n  UNSTABLE CONFIGS: {', '.join(failed)}")

    print()


def print_loss_curves(results, max_configs=6):
    """Print text-based loss curves for top configs."""
    print("\n" + "─" * 60)
    print("LOSS CURVES (train → val)")
    print("─" * 60)
    ranked = sorted(
        [(n, r) for n, r in results.items() if r["stable"]],
        key=lambda x: x[1]["val_mse_final"],
    )
    for name, r in ranked[:max_configs]:
        train_curve = " ".join(f"{v:.4f}" for v in r["train_losses"])
        val_curve = " ".join(f"{v:.4f}" for v in r["val_losses"])
        print(f"\n  {name}:")
        print(f"    train: {train_curve}")
        print(f"    val:   {val_curve}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Main Study Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_study(configs=None, epochs=EPOCHS, lr=LR, verbose=True):
    """
    Run the full ablation study.

    Parameters
    ----------
    configs : dict or None
        Override ABLATIONS with custom config dict. If None, uses default.
    epochs : int
        Training epochs per config.
    lr : float
        Learning rate.
    verbose : bool
        Print per-epoch progress.

    Returns
    -------
    OrderedDict of results per config.
    """
    if configs is None:
        configs = ABLATIONS

    print("=" * 80)
    print("  TFT End-to-End Ablation Study")
    print(f"  Synthetic Multi-Scale Wave Data (3 waves x 4 params -> 2 complex targets)")
    print(f"  seq_len={SEQ_LEN}, pred_len={PRED_LEN}, enc_in={ENC_IN}, c_out={C_OUT}")
    print(f"  d_model={D_MODEL}, n_heads={N_HEADS}, epochs={epochs}, lr={lr}")
    print(f"  train_samples={N_TRAIN}, val_samples={N_VAL}, noise_std={NOISE_STD}")
    print(f"  configs to test: {len(configs)}")
    print("=" * 80)

    # Generate data once (shared across configs)
    print("\nGenerating synthetic wave data...")
    train_loader, val_loader = make_dataloaders()
    print(f"  Train: {N_TRAIN} samples, {len(train_loader)} batches")
    print(f"  Val:   {N_VAL} samples, {len(val_loader)} batches")

    results = OrderedDict()
    total_t0 = time.time()

    for i, (name, overrides) in enumerate(configs.items()):
        print(f"\n{'─' * 70}")
        print(f"[{i+1}/{len(configs)}] {name}")
        print(f"  Overrides: {overrides if overrides else '(none — baseline)'}")
        print(f"{'─' * 70}")

        try:
            cfg = make_config(overrides)
            result = train_and_evaluate(name, cfg, train_loader, val_loader, epochs=epochs, lr=lr)
        except Exception as e:
            print(f"  *** ERROR: {e} ***")
            results[name] = {
                "name": name, "train_losses": [], "val_losses": [],
                "quantile_val_losses": [], "train_mse_0": float("nan"),
                "train_mse_final": float("nan"), "val_mse_final": float("nan"),
                "improvement_pct": 0.0, "generalization_gap_pct": float("inf"),
                "stable": False, "elapsed_s": 0.0, "n_params": 0,
                "model": None, "cfg": None, "error": str(e),
            }
            continue
        results[name] = result

        if verbose and result["stable"]:
            # Print first, middle, and last epoch
            tl = result["train_losses"]
            vl = result["val_losses"]
            mid = len(tl) // 2
            print(f"  Params: {result['n_params']:,d}")
            print(f"  Epoch  1: train={tl[0]:.5f}  val={vl[0]:.5f}")
            if mid > 0 and mid < len(tl) - 1:
                print(f"  Epoch {mid+1:2d}: train={tl[mid]:.5f}  val={vl[mid]:.5f}")
            print(f"  Epoch {len(tl):2d}: train={tl[-1]:.5f}  val={vl[-1]:.5f}")
            print(f"  Improvement: {result['improvement_pct']:.1f}%  |  "
                  f"Gen gap: {result['generalization_gap_pct']:.1f}%  |  "
                  f"Time: {result['elapsed_s']:.1f}s")
        elif not result["stable"]:
            print(f"  *** UNSTABLE — NaN/Inf detected ***")

    total_elapsed = time.time() - total_t0
    print(f"\nTotal study time: {total_elapsed:.1f}s")

    # Summary tables
    print_summary(results)
    print_loss_curves(results)

    # Interpretation for key configs
    interp_configs = ["baseline", "full_stack", "full_quantile"]
    for ic_name in interp_configs:
        if ic_name in results and results[ic_name]["stable"]:
            try:
                payload = run_interpretation(results[ic_name]["model"], val_loader)
                print_interpretation(payload, ic_name)
            except Exception as e:
                print(f"\n  Interpretation failed for '{ic_name}': {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: unittest integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestTFTE2E(unittest.TestCase):
    """Lightweight integration test — runs a subset of configs."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(SEED)
        cls.train_loader, cls.val_loader = make_dataloaders()

    def _run_config(self, name, overrides, epochs=8):
        cfg = make_config(overrides)
        result = train_and_evaluate(
            name, cfg, self.train_loader, self.val_loader, epochs=epochs, lr=LR,
        )
        self.assertTrue(result["stable"], f"{name} diverged (NaN/Inf)")
        self.assertGreater(
            result["improvement_pct"], 0,
            f"{name}: loss did not decrease (improvement={result['improvement_pct']:.1f}%)",
        )
        return result

    def test_baseline_learns(self):
        """Baseline LSTM TFT can learn the synthetic wave target."""
        self._run_config("baseline", {})

    def test_revin_helps_or_neutral(self):
        """RevIN does not degrade learning."""
        r = self._run_config("revin", {"tft_use_revin": True})
        self.assertGreater(r["improvement_pct"], 5, "RevIN should show meaningful learning")

    def test_fft_branch_learns(self):
        """FFT branch can learn multi-frequency data."""
        self._run_config("fft_branch", {"tft_use_fft_branch": True, "tft_fft_modes": 16})

    def test_full_stack_learns(self):
        """Full stack with all features can learn."""
        overrides = dict(ABLATIONS["full_stack"])
        self._run_config("full_stack", overrides, epochs=10)

    def test_full_quantile_learns(self):
        """Full stack + quantile head produces valid quantile predictions."""
        overrides = dict(ABLATIONS["full_quantile"])
        cfg = make_config(overrides)
        torch.manual_seed(SEED)
        model = tsl_tft.Model(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        mse_fn = nn.MSELoss()
        q_fn = QuantileLoss(cfg.tft_output_quantiles)
        model.train()
        for x_enc, x_mark_enc, x_dec, x_mark_dec, y in self.train_loader:
            optimizer.zero_grad()
            out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            pred = out[:, -cfg.pred_len:, :]
            loss = mse_fn(pred, y)
            qp = model.last_quantile_predictions
            self.assertIsNotNone(qp, "Quantile predictions should not be None")
            self.assertEqual(qp.shape[2], 3, "Should have 3 quantile levels")
            loss = loss + 0.5 * q_fn(qp, y)
            loss.backward()
            optimizer.step()
            break  # One batch is enough for shape validation

        # Check quantile ordering after a few steps of training
        model.train()
        for epoch in range(5):
            for x_enc, x_mark_enc, x_dec, x_mark_dec, y in self.train_loader:
                optimizer.zero_grad()
                out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                pred = out[:, -cfg.pred_len:, :]
                loss = mse_fn(pred, y) + 0.5 * q_fn(model.last_quantile_predictions, y)
                aux = getattr(model, "last_moe_aux_loss", None)
                if torch.is_tensor(aux):
                    loss = loss + cfg.tft_moe_aux_loss_coeff * aux
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        self.assertTrue(torch.isfinite(pred).all(), "Predictions should be finite after training")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--unittest" in sys.argv:
        sys.argv.remove("--unittest")
        unittest.main()
    else:
        results = run_ablation_study()
