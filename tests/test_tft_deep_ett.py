#!/usr/bin/env python
"""
Deep TFT ablation on real ETT data.

Runs multiple TFT configurations with deeper/wider models on ETTh1
to validate performance on real-world data with 17K+ samples.

Configs tested:
  1. baseline_shallow  - Current test config (d=128, 3 layers)
  2. baseline_deep     - Deeper (d=256, 4 layers, d_ff=512)
  3. deep_graph        - Deep + full graph features
  4. deep_full_stack   - Deep + all TFT features enabled
  5. deep_optimized    - Recommended production config

Usage:
  PYTHONPATH=. python tests/test_tft_deep_ett.py
  PYTHONPATH=. python tests/test_tft_deep_ett.py --configs baseline_shallow baseline_deep
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Configurations ──────────────────────────────────────────────────
CONFIGS = {
    "baseline_shallowest": {
        "d_model": 16, "n_heads": 2, "e_layers": 1, "d_ff": 32,
        "dropout": 0.2, "train_epochs": 50, "batch_size": 128,
        "learning_rate": 3e-4,
        # TFT features - all off
        "tft_temporal_backbone": "lstm",
        "tft_temporal_backbone_layers": 1,
        "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": False,
        "tft_graph_type": "dense",
        "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2,
        "tft_graph_temporal_evolution": False,
        "tft_graph_edge_features": False,
        "tft_per_target_heads": False,
        "tft_vsn_per_feature_gating": False,
        "tft_covariate_reattention": False,
        "tft_use_revin": True,  # RevIN is critical for ETTh1
        "tft_use_regime_moe": False,
        "tft_use_higher_order": False,
        "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False,
        "tft_full_attention": False,
        "tft_use_explicit_cross_attention": False,
    },
    # --- Advanced Features with small footprint ---
    "advanced_optimized": {
        "d_model": 32, "n_heads": 4, "e_layers": 2, "d_ff": 64,
        "dropout": 0.3, "train_epochs": 50, "batch_size": 128,
        "learning_rate": 3e-4,
        "tft_temporal_backbone": "hybrid_tcn_lstm",
        "tft_temporal_backbone_layers": 2,
        "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": True,
        "tft_graph_type": "temporal_sparse",
        "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2,
        "tft_graph_temporal_evolution": True,
        "tft_graph_edge_features": False,
        "tft_per_target_heads": True,
        "tft_vsn_per_feature_gating": True,
        "tft_covariate_reattention": True,
        "tft_use_revin": True,
        "tft_use_regime_moe": False,
        "tft_use_higher_order": False,
        "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": True,
        "tft_full_attention": False,
        "tft_use_explicit_cross_attention": True,
    },
    # --- Shallow baselines (same as synthetic test) ---
    "baseline_shallow": {
        "d_model": 128, "n_heads": 8, "e_layers": 3, "d_ff": 256,
        "dropout": 0.1, "train_epochs": 10, "batch_size": 256,
        "learning_rate": 5e-4,
        # TFT features - all off
        "tft_temporal_backbone": "hybrid_tcn_lstm",
        "tft_temporal_backbone_layers": 3,
        "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": False,
        "tft_graph_type": "dense",
        "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2,
        "tft_graph_temporal_evolution": False,
        "tft_graph_edge_features": False,
        "tft_per_target_heads": False,
        "tft_vsn_per_feature_gating": False,
        "tft_covariate_reattention": False,
        "tft_use_revin": False,
        "tft_use_regime_moe": False,
        "tft_use_higher_order": False,
        "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False,
        "tft_full_attention": False,
        "tft_use_explicit_cross_attention": False,
    },
    # --- Deep baselines ---
    "baseline_deep": {
        "d_model": 256, "n_heads": 8, "e_layers": 4, "d_ff": 512,
        "dropout": 0.15, "train_epochs": 10, "batch_size": 256,
        "learning_rate": 3e-4,
        "tft_temporal_backbone": "hybrid_tcn_lstm",
        "tft_temporal_backbone_layers": 4,
        "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": False,
        "tft_graph_type": "dense",
        "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2,
        "tft_graph_temporal_evolution": False,
        "tft_graph_edge_features": False,
        "tft_per_target_heads": False,
        "tft_vsn_per_feature_gating": False,
        "tft_covariate_reattention": False,
        "tft_use_revin": False,
        "tft_use_regime_moe": False,
        "tft_use_higher_order": False,
        "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False,
        "tft_full_attention": False,
        "tft_use_explicit_cross_attention": False,
    },
    # --- Deep + full graph ---
    "deep_graph": {
        "d_model": 256, "n_heads": 8, "e_layers": 4, "d_ff": 512,
        "dropout": 0.15, "train_epochs": 10, "batch_size": 256,
        "learning_rate": 3e-4,
        "tft_temporal_backbone": "hybrid_tcn_lstm",
        "tft_temporal_backbone_layers": 4,
        "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": True,
        "tft_graph_type": "temporal_sparse",
        "tft_graph_top_k": 5,
        "tft_graph_num_layers": 3,
        "tft_graph_temporal_evolution": True,
        "tft_graph_edge_features": True,
        "tft_per_target_heads": False,
        "tft_vsn_per_feature_gating": False,
        "tft_covariate_reattention": False,
        "tft_use_revin": False,
        "tft_use_regime_moe": False,
        "tft_use_higher_order": False,
        "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False,
        "tft_full_attention": False,
        "tft_use_explicit_cross_attention": False,
    },
    # --- Deep + all features ---
    "deep_full_stack": {
        "d_model": 256, "n_heads": 8, "e_layers": 4, "d_ff": 512,
        "dropout": 0.2, "train_epochs": 10, "batch_size": 32,
        "learning_rate": 2e-4,
        "tft_temporal_backbone": "hybrid_tcn_lstm",
        "tft_temporal_backbone_layers": 4,
        "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": True,
        "tft_graph_type": "temporal_sparse",
        "tft_graph_top_k": 5,
        "tft_graph_num_layers": 3,
        "tft_graph_temporal_evolution": True,
        "tft_graph_edge_features": True,
        "tft_per_target_heads": True,
        "tft_vsn_per_feature_gating": True,
        "tft_covariate_reattention": True,
        "tft_use_revin": True,
        "tft_use_regime_moe": True,
        "tft_use_higher_order": True,
        "tft_dual_attention_fusion": True,
        "tft_use_lag_attention": True,
        "tft_full_attention": True,
        "tft_use_explicit_cross_attention": True,
    },
    # --- Recommended production config ---
    "deep_optimized": {
        "d_model": 256, "n_heads": 8, "e_layers": 4, "d_ff": 512,
        "dropout": 0.15, "train_epochs": 10, "batch_size": 32,
        "learning_rate": 3e-4,
        "tft_temporal_backbone": "hybrid_tcn_lstm",
        "tft_temporal_backbone_layers": 4,
        "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": True,
        "tft_graph_type": "temporal_sparse",
        "tft_graph_top_k": 5,
        "tft_graph_num_layers": 3,
        "tft_graph_temporal_evolution": True,
        "tft_graph_edge_features": True,
        "tft_per_target_heads": True,
        "tft_vsn_per_feature_gating": False,
        "tft_covariate_reattention": True,
        "tft_use_revin": True,
        "tft_use_regime_moe": False,
        "tft_use_higher_order": False,
        "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": True,
        "tft_full_attention": False,
        "tft_use_explicit_cross_attention": True,
    },
    "large_40_patience": {
        "d_model": 256, "n_heads": 8, "e_layers": 4, "d_ff": 512,
        "dropout": 0.15, "train_epochs": 40, "batch_size": 64,
        "learning_rate": 3e-4, "patience": 8,
        "tft_temporal_backbone": "hybrid_tcn_lstm",
        "tft_temporal_backbone_layers": 4,
        "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": True,
        "tft_graph_type": "temporal_sparse",
        "tft_graph_top_k": 5,
        "tft_graph_num_layers": 3,
        "tft_graph_temporal_evolution": True,
        "tft_graph_edge_features": True,
        "tft_per_target_heads": True,
        "tft_vsn_per_feature_gating": True,
        "tft_covariate_reattention": True,
        "tft_use_revin": True,
        "tft_use_regime_moe": False,
        "tft_use_higher_order": False,
        "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": True,
        "tft_full_attention": False,
        "tft_use_explicit_cross_attention": True,
    },
    "balanced_30_epochs": {
        "d_model": 128, "n_heads": 8, "e_layers": 3, "d_ff": 256,
        "dropout": 0.25, "train_epochs": 30, "batch_size": 32,
        "learning_rate": 2e-4, "patience": 6,
        "tft_temporal_backbone": "hybrid_tcn_lstm",
        "tft_temporal_backbone_layers": 3,
        "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": True,
        "tft_graph_type": "temporal_sparse",
        "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2,
        "tft_graph_temporal_evolution": True,
        "tft_graph_edge_features": True,
        "tft_per_target_heads": True,
        "tft_vsn_per_feature_gating": True,
        "tft_covariate_reattention": True,
        "tft_use_revin": True,
        "tft_use_regime_moe": False,
        "tft_use_higher_order": False,
        "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": True,
        "tft_full_attention": False,
        "tft_use_explicit_cross_attention": True,
    },
    # --- Medium model with 20 epochs for validation loss testing ---
    "medium_20_epochs": {
        "d_model": 64, "n_heads": 4, "e_layers": 2, "d_ff": 128,
        "dropout": 0.35, "train_epochs": 20, "batch_size": 256,
        "learning_rate": 5e-4,
        "tft_temporal_backbone": "lstm",
        "tft_temporal_backbone_layers": 2,
        "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": False,
        "tft_graph_type": "dense",
        "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2,
        "tft_graph_temporal_evolution": False,
        "tft_graph_edge_features": False,
        "tft_per_target_heads": False,
        "tft_vsn_per_feature_gating": False,
        "tft_covariate_reattention": False,
        "tft_use_revin": True,
        "tft_use_regime_moe": False,
        "tft_use_higher_order": False,
        "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False,
        "tft_full_attention": False,
        "tft_use_explicit_cross_attention": False,
    },
}


def build_config(cfg_overrides):
    """Build a config namespace mimicking run.py args for TFT on ETTh1."""
    defaults = {
        # Task
        "task_name": "long_term_forecast",
        "is_training": 1,
        "model": "TemporalFusionTransformer",
        "model_id": "ETTh1_deep",
        # Data
        "data": "ETTh1",
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh1.csv",
        "features": "M",
        "target": "OT",
        "freq": "h",
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 96,
        "seasonal_patterns": "Monthly",
        "inverse": False,
        # Model
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "d_model": 128,
        "n_heads": 8,
        "e_layers": 3,
        "d_layers": 1,
        "d_ff": 256,
        "moving_avg": 25,
        "factor": 1,
        "distil": True,
        "dropout": 0.1,
        "embed": "timeF",
        "activation": "gelu",
        "channel_independence": 1,
        "decomp_method": "moving_avg",
        "use_norm": 1,
        "down_sampling_layers": 0,
        "down_sampling_window": 1,
        "down_sampling_method": None,
        "seg_len": 96,
        "top_k": 5,
        "num_kernels": 6,
        "expand": 2,
        "d_conv": 4,
        "patch_len": 16,
        "node_dim": 10,
        "gcn_depth": 2,
        "gcn_dropout": 0.3,
        "propalpha": 0.3,
        "conv_channel": 32,
        "skip_channel": 32,
        "individual": False,
        # Optimization
        "num_workers": 4,
        "itr": 1,
        "train_epochs": 30,
        "batch_size": 32,
        "patience": 5,
        "learning_rate": 5e-4,
        "des": "test",
        "loss": "MSE",
        "lradj": "type1",
        "use_amp": False,
        # GPU
        "use_gpu": True,
        "gpu": 0,
        "gpu_type": "cuda",
        "use_multi_gpu": False,
        "devices": "0",
        # Misc
        "checkpoints": "./checkpoints/",
        "mask_rate": 0.25,
        "anomaly_ratio": 0.25,
        "p_hidden_dims": [128, 128],
        "p_hidden_layers": 2,
        "use_dtw": False,
        "augmentation_ratio": 0,
        # TFT defaults
        "tft_observed_pos": None,
        "tft_static_pos": None,
        "tft_target_pos": None,
        "tft_use_swiglu": False,
        "tft_full_attention": False,
        "tft_cross_variable_mixing": False,
        "tft_allow_custom_known": False,
        "tft_vsn_residual_bypass": False,
        "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False,
        "tft_lag_scales": "1,2,4",
        "tft_temporal_backbone": "hybrid_tcn_lstm",
        "tft_temporal_backbone_layers": 3,
        "tft_temporal_kernel_size": 3,
        "tft_temporal_hidden_size": 0,
        "tft_use_higher_order": False,
        "tft_interaction_order": 2,
        "tft_interaction_rank": 0,
        "tft_use_regime_moe": False,
        "tft_use_explicit_cross_attention": False,
        "tft_cross_attention_type": "full",
        "tft_attention_position_bias": "none",
        "tft_attention_backend": "exact",
        "tft_rope_base": 10000.0,
        "tft_alibi_scale": 1.0,
        "tft_use_revin": False,
        "tft_revin_affine": False,
        "tft_use_quantile_head": False,
        "tft_output_quantiles": "0.1,0.5,0.9",
        "tft_num_regimes": 4,
        "tft_num_moe_experts": 4,
        "tft_moe_top_k": 2,
        "tft_moe_hidden_size": 0,
        "tft_moe_noise_epsilon": 1e-2,
        "tft_moe_aux_loss_coeff": 0.0,
        "tft_moe_capacity_factor": 1.25,
        "tft_per_target_heads": False,
        "tft_vsn_per_feature_gating": False,
        "tft_covariate_reattention": False,
        "tft_graph_type": "dense",
        "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2,
        "tft_graph_temporal_evolution": False,
        "tft_graph_edge_features": False,
        "tft_vsn_low_rank_threshold": 64,
    }
    defaults.update(cfg_overrides)

    class Config:
        pass

    cfg = Config()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_ett_config(name, cfg_overrides):
    """Train TFT on ETTh1 with given config overrides."""
    from models.TemporalFusionTransformer import Model
    from data_provider.data_factory import data_provider

    set_seed(SEED)
    cfg = build_config(cfg_overrides)
    cfg.model_id = f"ETTh1_deep_{name}"

    print(f"\n{'='*70}")
    print(f"  Config: {name}")
    print(f"  d_model={cfg.d_model}, e_layers={cfg.e_layers}, d_ff={cfg.d_ff}")
    print(f"  dropout={cfg.dropout}, lr={cfg.learning_rate}, epochs={cfg.train_epochs}")
    print(f"  graph={cfg.tft_graph_type}, revin={cfg.tft_use_revin}")
    print(f"{'='*70}")

    # Build model
    model = Model(cfg).to(DEVICE)
    n_params = count_params(model)
    print(f"  Parameters: {n_params:,}")

    # Data loaders
    train_set, train_loader = data_provider(cfg, "train")
    val_set, val_loader = data_provider(cfg, "val")
    test_set, test_loader = data_provider(cfg, "test")

    print(f"  Train samples: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    use_amp = DEVICE.type == "cuda"

    best_val = float("inf")
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 3
    train_losses = []
    val_losses = []
    t0 = time.time()

    for epoch in range(1, cfg.train_epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.float().to(DEVICE)
            batch_y = batch_y.float().to(DEVICE)
            batch_x_mark = batch_x_mark.float().to(DEVICE)
            batch_y_mark = batch_y_mark.float().to(DEVICE)

            # Decoder input: zeros for pred_len
            dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len:, :]).float().to(DEVICE)
            dec_inp = torch.cat([batch_y[:, :cfg.label_len, :], dec_inp], dim=1)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -cfg.pred_len:, :]
                targets = batch_y[:, -cfg.pred_len:, :]
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.float().to(DEVICE)
                batch_y = batch_y.float().to(DEVICE)
                batch_x_mark = batch_x_mark.float().to(DEVICE)
                batch_y_mark = batch_y_mark.float().to(DEVICE)

                dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len:, :]).float().to(DEVICE)
                dec_inp = torch.cat([batch_y[:, :cfg.label_len, :], dec_inp], dim=1)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs[:, -cfg.pred_len:, :]
                    targets = batch_y[:, -cfg.pred_len:, :]
                    val_loss += criterion(outputs, targets).item()
                n_val += 1

        val_loss = val_loss / max(n_val, 1)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        early_stop_patience = getattr(cfg, "patience", 3)
        if epoch % 2 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{cfg.train_epochs} | "
                  f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
                  f"Best: {best_val:.5f} | Time: {elapsed:.0f}s")

        if patience_counter >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

    # Load best model for testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ── Test ──
    model.eval()
    test_loss = 0.0
    n_test = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(DEVICE)
            batch_y = batch_y.float().to(DEVICE)
            batch_x_mark = batch_x_mark.float().to(DEVICE)
            batch_y_mark = batch_y_mark.float().to(DEVICE)

            dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len:, :]).float().to(DEVICE)
            dec_inp = torch.cat([batch_y[:, :cfg.label_len, :], dec_inp], dim=1)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -cfg.pred_len:, :]
                targets = batch_y[:, -cfg.pred_len:, :]
                test_loss += criterion(outputs, targets).item()
            n_test += 1

    test_loss = test_loss / max(n_test, 1)
    total_time = time.time() - t0

    # Gen gap
    gen_gap = (best_val - train_losses[-1]) / train_losses[-1] * 100 if train_losses[-1] > 0 else 0

    print(f"\n  ── Results: {name} ──")
    print(f"  Final Train MSE:  {train_losses[-1]:.5f}")
    print(f"  Best Val MSE:     {best_val:.5f}")
    print(f"  Test MSE:         {test_loss:.5f}")
    print(f"  Gen gap:          {gen_gap:.1f}%")
    print(f"  Params:           {n_params:,}")
    print(f"  Time:             {total_time:.1f}s")

    return {
        "name": name,
        "train_mse": train_losses[-1],
        "best_val_mse": best_val,
        "test_mse": test_loss,
        "gen_gap": gen_gap,
        "params": n_params,
        "time": total_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to run (default: all)")
    args = parser.parse_args()

    configs_to_run = args.configs or list(CONFIGS.keys())

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Configs to run: {configs_to_run}")
    print(f"Dataset: ETTh1 (17,420 rows, 7 features, hourly)")
    print(f"Task: long_term_forecast, seq_len=96, pred_len=96")

    results = []
    for name in configs_to_run:
        if name not in CONFIGS:
            print(f"WARNING: Unknown config '{name}', skipping")
            continue
        r = run_ett_config(name, CONFIGS[name])
        results.append(r)

    # ── Summary table ──
    if len(results) > 1:
        results.sort(key=lambda x: x["test_mse"])
        baseline_test = None
        for r in results:
            if "baseline" in r["name"]:
                baseline_test = r["test_mse"]
                break
        if baseline_test is None:
            baseline_test = results[-1]["test_mse"]

        print(f"\n{'='*90}")
        print(f"  FINAL RANKING (sorted by Test MSE)")
        print(f"{'='*90}")
        print(f"  {'Rank':<5} {'Config':<22} {'Val MSE':<12} {'Test MSE':<12} "
              f"{'vs Base':<10} {'Gen Gap':<10} {'Params':<12} {'Time':<8}")
        print(f"  {'-'*85}")
        for i, r in enumerate(results, 1):
            delta = (r["test_mse"] - baseline_test) / baseline_test * 100
            print(f"  {i:<5} {r['name']:<22} {r['best_val_mse']:<12.5f} {r['test_mse']:<12.5f} "
                  f"{delta:>+8.1f}%  {r['gen_gap']:>8.1f}%  {r['params']:>10,}  {r['time']:>6.0f}s")
        print(f"{'='*90}")


if __name__ == "__main__":
    main()
