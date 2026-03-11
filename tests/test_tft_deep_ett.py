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

if "MIOPEN_LOG_LEVEL" not in os.environ:
    os.environ["MIOPEN_LOG_LEVEL"] = "3"

if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import torch
import numpy as np
import random
from utils.tft_config import apply_tft_profile
from utils.tft_schema import resolve_target_positions, select_tft_truth

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _target_index_tensor(target_positions, device):
    return torch.as_tensor(tuple(target_positions), dtype=torch.long, device=device)


def _select_targets_for_loss(outputs, batch_y, pred_len, target_positions):
    if target_positions is None:
        pred = outputs[:, -pred_len:, :]
        true = batch_y[:, -pred_len:, :]
        return pred, true

    pred = outputs[:, -pred_len:, :]
    true = select_tft_truth(batch_y, pred_len, target_positions)
    if pred.shape != true.shape:
        raise RuntimeError(
            f"TFT prediction/target shape mismatch in deep_ett harness: pred={tuple(pred.shape)} true={tuple(true.shape)}."
        )
    return pred, true


def _naive_prediction(batch_x, pred_len, target_positions):
    if target_positions is None:
        return batch_x.mean(1, keepdim=True).expand(-1, pred_len, -1)
    index = _target_index_tensor(target_positions, device=batch_x.device)
    selected = batch_x.index_select(-1, index)
    return selected.mean(1, keepdim=True).expand(-1, pred_len, -1)


# ── Configurations ──────────────────────────────────────────────────
# ETTh1 training set: 8,449 samples.
# Param budget rule: target params/samples < 10× for reasonable generalisation.
# Actual param counts (verified):
#   d=8,  e=1, lstm       →   12K  (1.5×)
#   d=12, e=1, lstm       →   26K  (3.1×)
#   d=16, e=1, lstm       →   55K  (6.5×)   ← safe baseline
#   d=24, e=1, lstm       →  142K  (16.8×)  ← borderline
#   d=32, e=1, lstm       →  286K  (33.9×)  ← overfit risk without heavy dropout
#   d=64, e=2, lstm       →  819K  (97×)    ← was baseline_shallow — way too large
#
# Training schedule keys (all optional per-config):
#   patience        : early-stop after N epochs of no val improvement. None = disabled.
#   warmup_epochs   : linear LR ramp from lr/warmup_epochs to lr. Default 5.
#   plateau_patience: ReduceLROnPlateau patience. Default 5.
#   plateau_factor  : LR multiplier on plateau. Default 0.5.
#   min_lr          : floor for plateau decay. Default 1e-6.
CONFIGS = {
    # ── TIER 1: Safe baselines (< 10× params/samples) ───────────────────
    # 55K params / 8449 samples = 6.5×  ← the honest baseline for ETTh1
    "baseline_safe": {
        "d_model": 16, "n_heads": 2, "e_layers": 1, "d_ff": 32,
        "dropout": 0.2, "train_epochs": 150, "batch_size": 64,
        "learning_rate": 5e-4,
        "patience": None, "warmup_epochs": 5, "plateau_patience": 6, "plateau_factor": 0.5, "min_lr": 5e-7,
        "tft_temporal_backbone": "lstm", "tft_temporal_backbone_layers": 1, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": False, "tft_graph_type": "dense", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2, "tft_graph_temporal_evolution": False, "tft_graph_edge_features": False,
        "tft_per_target_heads": False, "tft_vsn_per_feature_gating": False, "tft_covariate_reattention": False,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False, "tft_full_attention": False, "tft_use_explicit_cross_attention": False,
    },
    # 26K params = 3.1× — minimal capacity, check model can learn at all
    "micro": {
        "d_model": 12, "n_heads": 2, "e_layers": 1, "d_ff": 24,
        "dropout": 0.1, "train_epochs": 150, "batch_size": 64,
        "learning_rate": 5e-4,
        "patience": None, "warmup_epochs": 5, "plateau_patience": 6, "plateau_factor": 0.5, "min_lr": 5e-7,
        "tft_temporal_backbone": "lstm", "tft_temporal_backbone_layers": 1, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": False, "tft_graph_type": "dense", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2, "tft_graph_temporal_evolution": False, "tft_graph_edge_features": False,
        "tft_per_target_heads": False, "tft_vsn_per_feature_gating": False, "tft_covariate_reattention": False,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False, "tft_full_attention": False, "tft_use_explicit_cross_attention": False,
    },
    # 55K-ish effective native TFT capacity with the upgraded regularisation path:
    # real attention-probability dropout, active early stopping, and gentler LR.
    # This is the recommended first rerun when val loss previously rose after warmup.
    "ett_stable_v2": {
        "tft_profile": "extended_safe",
        "d_model": 16, "n_heads": 2, "e_layers": 1, "d_ff": 32,
        "dropout": 0.25, "tft_attention_dropout": 0.10,
        "train_epochs": 80, "batch_size": 64,
        "learning_rate": 3e-4,
        "patience": 12, "warmup_epochs": 8, "plateau_patience": 4, "plateau_factor": 0.5, "min_lr": 1e-6,
        "tft_temporal_backbone": "lstm", "tft_temporal_backbone_layers": 1, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": False, "tft_graph_type": "dense", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2, "tft_graph_temporal_evolution": False, "tft_graph_edge_features": False,
        "tft_per_target_heads": False, "tft_vsn_per_feature_gating": False, "tft_covariate_reattention": False,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False, "tft_full_attention": False, "tft_use_explicit_cross_attention": False,
        "tft_stochastic_depth_rate": 0.0, "tft_payload_stack_layers": False,
    },
    # Single-target OT forecasting through the same native TFT path.
    # This aligns better with the usual "predict OT" expectation than the
    # default 7-channel multivariate-to-multivariate harness objective.
    "ett_ot_stable_v1": {
        "tft_profile": "extended_safe",
        "features": "MS",
        "enc_in": 7, "dec_in": 7, "c_out": 1,
        "tft_target_pos": [6],
        "d_model": 16, "n_heads": 2, "e_layers": 1, "d_ff": 32,
        "dropout": 0.20, "tft_attention_dropout": 0.08,
        "train_epochs": 80, "batch_size": 64,
        "learning_rate": 2e-4,
        "patience": 10, "warmup_epochs": 8, "plateau_patience": 4, "plateau_factor": 0.5, "min_lr": 1e-6,
        "tft_temporal_backbone": "lstm", "tft_temporal_backbone_layers": 1, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": False, "tft_graph_type": "dense", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2, "tft_graph_temporal_evolution": False, "tft_graph_edge_features": False,
        "tft_per_target_heads": False, "tft_vsn_per_feature_gating": False, "tft_covariate_reattention": False,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False, "tft_full_attention": False, "tft_use_explicit_cross_attention": False,
        "tft_stochastic_depth_rate": 0.0, "tft_payload_stack_layers": False,
    },
    # Same OT-only objective, but with a much gentler LR ceiling because the
    # stable_v1 curve peaked before warmup completed and degraded once LR rose.
    "ett_ot_low_lr_v1": {
        "tft_profile": "extended_safe",
        "features": "MS",
        "enc_in": 7, "dec_in": 7, "c_out": 1,
        "tft_target_pos": [6],
        "d_model": 16, "n_heads": 2, "e_layers": 1, "d_ff": 32,
        "dropout": 0.20, "tft_attention_dropout": 0.08,
        "train_epochs": 80, "batch_size": 64,
        "learning_rate": 1e-4,
        "patience": 10, "warmup_epochs": 6, "plateau_patience": 3, "plateau_factor": 0.5, "min_lr": 5e-7,
        "weight_decay": 5e-3,
        "tft_temporal_backbone": "lstm", "tft_temporal_backbone_layers": 1, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": False, "tft_graph_type": "dense", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2, "tft_graph_temporal_evolution": False, "tft_graph_edge_features": False,
        "tft_per_target_heads": False, "tft_vsn_per_feature_gating": False, "tft_covariate_reattention": False,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False, "tft_full_attention": False, "tft_use_explicit_cross_attention": False,
        "tft_stochastic_depth_rate": 0.0, "tft_payload_stack_layers": False,
    },
    # Same as low_lr_v1, but removes RevIN in case the extra per-window
    # normalization/denormalization is hurting OT-only stability on ETTh1.
    "ett_ot_no_revin_v1": {
        "tft_profile": "extended_safe",
        "features": "MS",
        "enc_in": 7, "dec_in": 7, "c_out": 1,
        "tft_target_pos": [6],
        "d_model": 16, "n_heads": 2, "e_layers": 1, "d_ff": 32,
        "dropout": 0.20, "tft_attention_dropout": 0.08,
        "train_epochs": 80, "batch_size": 64,
        "learning_rate": 1e-4,
        "patience": 10, "warmup_epochs": 6, "plateau_patience": 3, "plateau_factor": 0.5, "min_lr": 5e-7,
        "weight_decay": 5e-3,
        "tft_temporal_backbone": "lstm", "tft_temporal_backbone_layers": 1, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": False, "tft_graph_type": "dense", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2, "tft_graph_temporal_evolution": False, "tft_graph_edge_features": False,
        "tft_per_target_heads": False, "tft_vsn_per_feature_gating": False, "tft_covariate_reattention": False,
        "tft_use_revin": False, "tft_revin_affine": False,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False, "tft_full_attention": False, "tft_use_explicit_cross_attention": False,
        "tft_stochastic_depth_rate": 0.0, "tft_payload_stack_layers": False,
    },
    # ── TIER 2: Borderline (10-30×) — acceptable with high dropout + AdamW ─
    # 142K params = 16.8× — needs dropout ≥ 0.3 to generalise
    "small_features": {
        "d_model": 24, "n_heads": 4, "e_layers": 1, "d_ff": 48,
        "dropout": 0.35, "train_epochs": 120, "batch_size": 64,
        "learning_rate": 3e-4,
        "patience": None, "warmup_epochs": 5, "plateau_patience": 6, "plateau_factor": 0.5, "min_lr": 5e-7,
        "tft_temporal_backbone": "lstm", "tft_temporal_backbone_layers": 1, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": False, "tft_graph_type": "dense", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2, "tft_graph_temporal_evolution": False, "tft_graph_edge_features": False,
        "tft_per_target_heads": False, "tft_vsn_per_feature_gating": False, "tft_covariate_reattention": False,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False, "tft_full_attention": False, "tft_use_explicit_cross_attention": False,
    },
    # 286K params = 33.9× — needs heavy dropout + AdamW weight_decay to stay honest
    "small_features_advanced": {
        "d_model": 32, "n_heads": 4, "e_layers": 1, "d_ff": 64,
        "dropout": 0.45, "train_epochs": 120, "batch_size": 64,
        "learning_rate": 2e-4,
        "patience": None, "warmup_epochs": 5, "plateau_patience": 6, "plateau_factor": 0.5, "min_lr": 5e-7,
        "tft_temporal_backbone": "hybrid_tcn_lstm", "tft_temporal_backbone_layers": 2, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": True, "tft_graph_type": "temporal_sparse", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2, "tft_graph_temporal_evolution": False, "tft_graph_edge_features": False,
        "tft_per_target_heads": True, "tft_vsn_per_feature_gating": False, "tft_covariate_reattention": False,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": True, "tft_full_attention": False, "tft_use_explicit_cross_attention": False,
    },
    # ── TIER 3: Overparameterised — require stochastic depth + high dropout ──
    # These are the configs from the original test — kept for ablation, not
    # expected to generalise well on 8K samples without the full ETT (172K rows).
    "baseline_shallow": {
        "d_model": 64, "n_heads": 4, "e_layers": 2, "d_ff": 128,
        # 819K params = 97× — needs very high dropout + stochastic depth
        "dropout": 0.5, "train_epochs": 100, "batch_size": 128,
        "learning_rate": 1e-4,
        "patience": None, "warmup_epochs": 5, "plateau_patience": 6, "plateau_factor": 0.5, "min_lr": 5e-7,
        "tft_temporal_backbone": "lstm", "tft_temporal_backbone_layers": 1, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": False, "tft_graph_type": "dense", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2, "tft_graph_temporal_evolution": False, "tft_graph_edge_features": False,
        "tft_per_target_heads": False, "tft_vsn_per_feature_gating": False, "tft_covariate_reattention": False,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False, "tft_full_attention": False, "tft_use_explicit_cross_attention": False,
        "tft_stochastic_depth_rate": 0.1,   # layer-drop as extra regulariser
    },
    "baseline_deep": {
        "d_model": 256, "n_heads": 8, "e_layers": 4, "d_ff": 512,
        "dropout": 0.5, "train_epochs": 80, "batch_size": 128,
        "learning_rate": 5e-5,
        "patience": None, "warmup_epochs": 10, "plateau_patience": 6, "plateau_factor": 0.5, "min_lr": 1e-7,
        "tft_temporal_backbone": "hybrid_tcn_lstm", "tft_temporal_backbone_layers": 4, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": False, "tft_graph_type": "dense", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2, "tft_graph_temporal_evolution": False, "tft_graph_edge_features": False,
        "tft_per_target_heads": False, "tft_vsn_per_feature_gating": False, "tft_covariate_reattention": False,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False, "tft_full_attention": False, "tft_use_explicit_cross_attention": False,
        "tft_stochastic_depth_rate": 0.2,
    },
    "deep_graph": {
        "d_model": 256, "n_heads": 8, "e_layers": 4, "d_ff": 512,
        "dropout": 0.5, "train_epochs": 80, "batch_size": 128,
        "learning_rate": 5e-5,
        "patience": None, "warmup_epochs": 10, "plateau_patience": 6, "plateau_factor": 0.5, "min_lr": 1e-7,
        "tft_temporal_backbone": "hybrid_tcn_lstm", "tft_temporal_backbone_layers": 4, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": True, "tft_graph_type": "temporal_sparse", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 3, "tft_graph_temporal_evolution": True, "tft_graph_edge_features": True,
        "tft_per_target_heads": False, "tft_vsn_per_feature_gating": False, "tft_covariate_reattention": False,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False, "tft_full_attention": False, "tft_use_explicit_cross_attention": False,
        "tft_stochastic_depth_rate": 0.2,
    },
    "deep_full_stack": {
        "d_model": 256, "n_heads": 8, "e_layers": 4, "d_ff": 512,
        "dropout": 0.5, "train_epochs": 80, "batch_size": 64,
        "learning_rate": 3e-5,
        "patience": None, "warmup_epochs": 10, "plateau_patience": 6, "plateau_factor": 0.5, "min_lr": 1e-7,
        "tft_temporal_backbone": "hybrid_tcn_lstm", "tft_temporal_backbone_layers": 4, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": True, "tft_graph_type": "temporal_sparse", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 3, "tft_graph_temporal_evolution": True, "tft_graph_edge_features": True,
        "tft_per_target_heads": True, "tft_vsn_per_feature_gating": True, "tft_covariate_reattention": True,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": True, "tft_use_higher_order": True, "tft_dual_attention_fusion": True,
        "tft_use_lag_attention": True, "tft_full_attention": True, "tft_use_explicit_cross_attention": True,
        "tft_stochastic_depth_rate": 0.2,
    },
    "deep_optimized": {
        "d_model": 256, "n_heads": 8, "e_layers": 4, "d_ff": 512,
        "dropout": 0.5, "train_epochs": 80, "batch_size": 64,
        "learning_rate": 5e-5,
        "patience": None, "warmup_epochs": 10, "plateau_patience": 6, "plateau_factor": 0.5, "min_lr": 1e-7,
        "tft_temporal_backbone": "hybrid_tcn_lstm", "tft_temporal_backbone_layers": 4, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": True, "tft_graph_type": "temporal_sparse", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 3, "tft_graph_temporal_evolution": True, "tft_graph_edge_features": True,
        "tft_per_target_heads": True, "tft_vsn_per_feature_gating": False, "tft_covariate_reattention": True,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": True, "tft_full_attention": False, "tft_use_explicit_cross_attention": True,
        "tft_stochastic_depth_rate": 0.2,
    },
    "large_40_patience": {
        "d_model": 256, "n_heads": 8, "e_layers": 4, "d_ff": 512,
        "dropout": 0.5, "train_epochs": 80, "batch_size": 64,
        "learning_rate": 5e-5,
        "patience": None, "warmup_epochs": 10, "plateau_patience": 6, "plateau_factor": 0.5, "min_lr": 1e-7,
        "tft_temporal_backbone": "hybrid_tcn_lstm", "tft_temporal_backbone_layers": 4, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": True, "tft_graph_type": "temporal_sparse", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 3, "tft_graph_temporal_evolution": True, "tft_graph_edge_features": True,
        "tft_per_target_heads": True, "tft_vsn_per_feature_gating": True, "tft_covariate_reattention": True,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": True, "tft_full_attention": False, "tft_use_explicit_cross_attention": True,
        "tft_stochastic_depth_rate": 0.2,
    },
    "balanced_30_epochs": {
        "d_model": 128, "n_heads": 8, "e_layers": 3, "d_ff": 256,
        "dropout": 0.5, "train_epochs": 80, "batch_size": 64,
        "learning_rate": 5e-5,
        "patience": None, "warmup_epochs": 8, "plateau_patience": 6, "plateau_factor": 0.5, "min_lr": 1e-7,
        "tft_temporal_backbone": "hybrid_tcn_lstm", "tft_temporal_backbone_layers": 3, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": True, "tft_graph_type": "temporal_sparse", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2, "tft_graph_temporal_evolution": True, "tft_graph_edge_features": True,
        "tft_per_target_heads": True, "tft_vsn_per_feature_gating": True, "tft_covariate_reattention": True,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": True, "tft_full_attention": False, "tft_use_explicit_cross_attention": True,
        "tft_stochastic_depth_rate": 0.15,
    },
    "medium_20_epochs": {
        "d_model": 64, "n_heads": 4, "e_layers": 2, "d_ff": 128,
        "dropout": 0.5, "train_epochs": 100, "batch_size": 128,
        "learning_rate": 1e-4,
        "patience": None, "warmup_epochs": 6, "plateau_patience": 6, "plateau_factor": 0.5, "min_lr": 5e-7,
        "tft_temporal_backbone": "lstm", "tft_temporal_backbone_layers": 2, "tft_temporal_kernel_size": 3,
        "tft_cross_variable_mixing": False, "tft_graph_type": "dense", "tft_graph_top_k": 5,
        "tft_graph_num_layers": 2, "tft_graph_temporal_evolution": False, "tft_graph_edge_features": False,
        "tft_per_target_heads": False, "tft_vsn_per_feature_gating": False, "tft_covariate_reattention": False,
        "tft_use_revin": True, "tft_revin_affine": True,
        "tft_use_regime_moe": False, "tft_use_higher_order": False, "tft_dual_attention_fusion": False,
        "tft_use_lag_attention": False, "tft_full_attention": False, "tft_use_explicit_cross_attention": False,
        "tft_stochastic_depth_rate": 0.1,
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
        "num_workers": 0,
        "itr": 1,
        "train_epochs": 30,
        "batch_size": 32,
        "patience": 5,
        "learning_rate": 5e-4,
        "des": "test",
        "loss": "MSE",
        "lradj": "type1",
        "use_amp": False,
        "weight_decay": 1e-2,
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
        "tft_stochastic_depth_rate": 0.0,
        "tft_gradient_checkpointing": False,
        "tft_payload_stack_layers": True,
        "tft_vsn_n_selection_heads": 1,
    }
    defaults.update(cfg_overrides)

    class Config:
        pass

    cfg = Config()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return apply_tft_profile(cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_ett_config(name, cfg_overrides, *, max_train_batches=None, max_eval_batches=None, use_amp=False):
    """Train TFT on ETTh1 with given config overrides."""
    from models.TemporalFusionTransformer import Model
    from data_provider.data_factory import data_provider

    set_seed(SEED)
    cfg = build_config(cfg_overrides)
    cfg.model_id = f"ETTh1_deep_{name}"
    target_positions = resolve_target_positions(cfg)

    # Per-config training knobs with sensible defaults
    early_stop_patience = getattr(cfg, "patience", None)  # None = disabled
    warmup_epochs       = getattr(cfg, "warmup_epochs", 5)
    plateau_patience    = getattr(cfg, "plateau_patience", 4)
    plateau_factor      = getattr(cfg, "plateau_factor", 0.5)
    min_lr              = getattr(cfg, "min_lr", 1e-6)

    print(f"\n{'='*70}")
    print(f"  Config: {name}")
    print(f"  d_model={cfg.d_model}, e_layers={cfg.e_layers}, d_ff={cfg.d_ff}")
    print(f"  dropout={cfg.dropout}, lr={cfg.learning_rate}, epochs={cfg.train_epochs}")
    print(f"  revin={cfg.tft_use_revin}, graph={cfg.tft_graph_type}")
    print(f"  weight_decay={cfg.weight_decay}")
    print(f"  warmup={warmup_epochs} epochs | plateau_patience={plateau_patience} | early_stop={'disabled' if early_stop_patience is None else early_stop_patience}")
    print(f"  use_amp={use_amp} | max_train_batches={max_train_batches} | max_eval_batches={max_eval_batches}")
    print(f"{'='*70}")

    # Build model
    model = Model(cfg).to(DEVICE)
    n_params = count_params(model)
    ratio = n_params / 8449
    print(f"  Parameters: {n_params:,}  ({ratio:.1f}× train samples)")

    # Data loaders
    train_set, train_loader = data_provider(cfg, "train")
    val_set, val_loader = data_provider(cfg, "val")
    test_set, test_loader = data_provider(cfg, "test")

    print(f"  Train samples: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # Compute naive baseline MSE on val and test (predict mean of x_enc window).
    # This is the natural floor for val/test — it differs from train because
    # ETTh1's val period has higher within-window variance (later time, more drift).
    # The meaningful metric is how much the model beats this baseline, not raw val MSE.
    import torch as _torch
    naive_mse = {}
    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        _mse = 0; _n = 0
        with _torch.no_grad():
            for bx, by, _, _ in loader:
                tgt = select_tft_truth(by, cfg.pred_len, target_positions)
                pred = _naive_prediction(bx, cfg.pred_len, target_positions)
                _mse += ((pred - tgt) ** 2).mean().item(); _n += 1
        naive_mse[split_name] = _mse / max(_n, 1)
    print(f"  Naive mean-predictor MSE — train: {naive_mse['train']:.4f}  "
          f"val: {naive_mse['val']:.4f}  test: {naive_mse['test']:.4f}")

    # ── Optimizer ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=float(getattr(cfg, "weight_decay", 1e-2)),
        betas=(0.9, 0.98),
    )
    criterion = torch.nn.MSELoss()

    # ── LR Schedule: linear warmup → ReduceLROnPlateau ──────────────
    # Warmup: ramp from lr/10 to lr over `warmup_epochs` epochs.
    # Without warmup, large initial gradients from random weights push the
    # model into a sharp, poorly-generalising minimum in the first 1-2 epochs.
    base_lr = cfg.learning_rate

    def warmup_lambda(epoch):
        # epoch is 0-indexed here (PyTorch scheduler calls after optimizer step)
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # 1/W, 2/W, ..., W/W = 1.0
        return 1.0  # ReduceLROnPlateau takes over after warmup

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=plateau_patience, factor=plateau_factor,
        min_lr=min_lr, verbose=False,
    )

    best_val = float("inf")
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    t0 = time.time()

    for epoch in range(1, cfg.train_epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            batch_x      = batch_x.float().to(DEVICE)
            batch_y      = batch_y.float().to(DEVICE)
            batch_x_mark = batch_x_mark.float().to(DEVICE)
            batch_y_mark = batch_y_mark.float().to(DEVICE)

            dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len:, :]).to(DEVICE)
            dec_inp = torch.cat([batch_y[:, :cfg.label_len, :], dec_inp], dim=1)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred, true = _select_targets_for_loss(outputs, batch_y, cfg.pred_len, target_positions)
            loss = criterion(pred, true)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1
            if max_train_batches is not None and (batch_idx + 1) >= max_train_batches:
                break

        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        # Warmup step (epoch-level, runs every epoch during warmup phase)
        if epoch <= warmup_epochs:
            warmup_scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                batch_x      = batch_x.float().to(DEVICE)
                batch_y      = batch_y.float().to(DEVICE)
                batch_x_mark = batch_x_mark.float().to(DEVICE)
                batch_y_mark = batch_y_mark.float().to(DEVICE)

                dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len:, :]).to(DEVICE)
                dec_inp = torch.cat([batch_y[:, :cfg.label_len, :], dec_inp], dim=1)

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred, true = _select_targets_for_loss(outputs, batch_y, cfg.pred_len, target_positions)
                val_loss += criterion(pred, true).item()
                n_val    += 1
                if max_eval_batches is not None and (batch_idx + 1) >= max_eval_batches:
                    break

        val_loss = val_loss / max(n_val, 1)
        val_losses.append(val_loss)

        # ReduceLROnPlateau kicks in after warmup
        if epoch > warmup_epochs:
            plateau_scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        if val_loss < best_val:
            best_val         = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 2 == 0 or epoch == 1:
            elapsed = time.time() - t0
            warmup_tag = " [warmup]" if epoch <= warmup_epochs else ""
            beat_naive = (naive_mse['val'] - val_loss) / naive_mse['val'] * 100
            print(f"  Epoch {epoch:3d}/{cfg.train_epochs} | "
                  f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
                  f"Best: {best_val:.5f} | vs naive: {beat_naive:+.1f}% | "
                  f"LR: {current_lr:.2e}{warmup_tag} | {elapsed:.0f}s")

        # Early stopping — only if patience is configured (not None)
        if early_stop_patience is not None and patience_counter >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch} (no val improvement for {early_stop_patience} epochs)")
            break

    # Load best model for testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ── Test ──
    model.eval()
    test_loss = 0.0
    n_test = 0
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(DEVICE)
            batch_y = batch_y.float().to(DEVICE)
            batch_x_mark = batch_x_mark.float().to(DEVICE)
            batch_y_mark = batch_y_mark.float().to(DEVICE)

            dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len:, :]).float().to(DEVICE)
            dec_inp = torch.cat([batch_y[:, :cfg.label_len, :], dec_inp], dim=1)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred, true = _select_targets_for_loss(outputs, batch_y, cfg.pred_len, target_positions)
                test_loss += criterion(pred, true).item()
            n_test += 1
            if max_eval_batches is not None and (batch_idx + 1) >= max_eval_batches:
                break

    test_loss = test_loss / max(n_test, 1)
    total_time = time.time() - t0

    # Gen gap
    gen_gap = (best_val - train_losses[-1]) / train_losses[-1] * 100 if train_losses[-1] > 0 else 0
    beat_naive_val  = (naive_mse['val']  - best_val)  / naive_mse['val']  * 100
    beat_naive_test = (naive_mse['test'] - test_loss)  / naive_mse['test'] * 100

    print(f"\n  ── Results: {name} ──")
    print(f"  Naive baseline MSE — val: {naive_mse['val']:.4f}  test: {naive_mse['test']:.4f}")
    print(f"  Best Val MSE:   {best_val:.5f}  (beats naive by {beat_naive_val:+.1f}%)")
    print(f"  Test MSE:       {test_loss:.5f}  (beats naive by {beat_naive_test:+.1f}%)")
    print(f"  Final Train MSE:{train_losses[-1]:.5f}")
    print(f"  Params:         {n_params:,}  ({ratio:.1f}×)")
    print(f"  Time:           {total_time:.1f}s")

    return {
        "name": name,
        "train_mse": train_losses[-1],
        "best_val_mse": best_val,
        "val_naive": naive_mse['val'],
        "beat_naive_val": beat_naive_val,
        "test_mse": test_loss,
        "beat_naive_test": beat_naive_test,
        "params": n_params,
        "time": total_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific configs to run (default: all)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override every selected config's train_epochs for quicker benchmark runs.")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Stop each training epoch after this many batches.")
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="Stop each validation/test pass after this many batches.")
    parser.add_argument("--use-amp", action="store_true", default=False,
                        help="Enable CUDA AMP for the evaluation pass.")
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
        cfg_overrides = dict(CONFIGS[name])
        if args.epochs is not None:
            cfg_overrides["train_epochs"] = args.epochs
        r = run_ett_config(
            name,
            cfg_overrides,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            use_amp=args.use_amp,
        )
        results.append(r)

    # ── Summary table ──
    if len(results) > 1:
        results.sort(key=lambda x: x["beat_naive_val"], reverse=True)
        print(f"\n{'='*100}")
        print(f"  FINAL RANKING — sorted by how much model beats naive mean-predictor on val")
        print(f"  Naive baseline: predict mean(x_enc) for all future steps")
        print(f"  Val naive MSE is naturally higher than train (later period, more drift in ETTh1)")
        print(f"{'='*100}")
        print(f"  {'Rank':<5} {'Config':<24} {'Val MSE':<10} {'Val Naive':<11} {'Beat%↑':<10} "
              f"{'Test MSE':<10} {'Params':<12} {'Time':<8}")
        print(f"  {'-'*95}")
        for i, r in enumerate(results, 1):
            print(f"  {i:<5} {r['name']:<24} {r['best_val_mse']:<10.5f} "
                  f"{r['val_naive']:<11.4f} {r['beat_naive_val']:>+8.1f}%  "
                  f"{r['test_mse']:<10.5f} {r['params']:>10,}  {r['time']:>6.0f}s")
        print(f"{'='*100}")


if __name__ == "__main__":
    main()
