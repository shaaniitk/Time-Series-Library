#!/usr/bin/env python
"""
Controlled TFT comparison: CustomTFT vs NixtlaTFT on ETTh1.
Identical train/val/test splits, identical optimizer, identical seeds.
Purpose: determine whether the high validation loss in CustomTFT is a
model bug or a configuration / overfitting issue.
"""
import os, sys, time, random
os.environ["MIOPEN_LOG_LEVEL"] = "3"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import torch
import numpy as np

sys.path.insert(0, "/home/kalki/Documents/workspace/Time-Series-Library")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ── shared config ────────────────────────────────────────────────────
SHARED = {
    # task
    "task_name": "long_term_forecast", "is_training": 1,
    "data": "ETTh1", "root_path": "./dataset/ETT-small/",
    "data_path": "ETTh1.csv", "features": "M", "target": "OT", "freq": "h",
    "seq_len": 96, "label_len": 48, "pred_len": 96,
    "enc_in": 7, "dec_in": 7, "c_out": 7,
    # model size — same for both
    "d_model": 64, "n_heads": 4, "e_layers": 2, "d_layers": 1, "d_ff": 128,
    "dropout": 0.3,
    # optimization — same for both
    "learning_rate": 3e-4, "train_epochs": 30, "batch_size": 256, "patience": 7,
    # misc TSL fields needed by data_provider
    "embed": "timeF", "activation": "gelu",
    "channel_independence": 1, "decomp_method": "moving_avg",
    "use_norm": 1, "down_sampling_layers": 0, "down_sampling_window": 1,
    "down_sampling_method": None, "seg_len": 96, "top_k": 5,
    "num_kernels": 6, "expand": 2, "d_conv": 4, "patch_len": 16,
    "node_dim": 10, "gcn_depth": 2, "gcn_dropout": 0.3,
    "propalpha": 0.3, "conv_channel": 32, "skip_channel": 32,
    "individual": False, "moving_avg": 25, "factor": 1, "distil": True,
    "seasonal_patterns": "Monthly", "inverse": False,
    "num_workers": 0, "itr": 1, "des": "test", "loss": "MSE",
    "lradj": "type1", "use_amp": False, "use_gpu": True, "gpu": 0,
    "gpu_type": "cuda", "use_multi_gpu": False, "devices": "0",
    "checkpoints": "./checkpoints/", "mask_rate": 0.25,
    "anomaly_ratio": 0.25, "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2, "use_dtw": False, "augmentation_ratio": 0,
    # TFT defaults
    "tft_observed_pos": None, "tft_static_pos": None, "tft_target_pos": None,
    "tft_use_swiglu": False, "tft_full_attention": False,
    "tft_cross_variable_mixing": False, "tft_allow_custom_known": False,
    "tft_vsn_residual_bypass": True, "tft_dual_attention_fusion": False,
    "tft_use_lag_attention": False, "tft_lag_scales": "1,2,4",
    "tft_temporal_backbone": "lstm",          # simplest backbone
    "tft_temporal_backbone_layers": 1,
    "tft_temporal_kernel_size": 3, "tft_temporal_hidden_size": 0,
    "tft_use_higher_order": False, "tft_interaction_order": 2,
    "tft_interaction_rank": 0, "tft_use_regime_moe": False,
    "tft_use_explicit_cross_attention": False,
    "tft_cross_attention_type": "full",
    "tft_attention_position_bias": "none", "tft_attention_backend": "exact",
    "tft_rope_base": 10000.0, "tft_alibi_scale": 1.0,
    "tft_use_revin": True,                    # RevIN on
    "tft_revin_affine": True,
    "tft_use_quantile_head": False, "tft_output_quantiles": "0.1,0.5,0.9",
    "tft_num_regimes": 4, "tft_num_moe_experts": 4, "tft_moe_top_k": 2,
    "tft_moe_hidden_size": 0, "tft_moe_noise_epsilon": 1e-2,
    "tft_moe_aux_loss_coeff": 0.0, "tft_moe_capacity_factor": 1.25,
    "tft_per_target_heads": False, "tft_vsn_per_feature_gating": False,
    "tft_covariate_reattention": False, "tft_graph_type": "dense",
    "tft_graph_top_k": 5, "tft_graph_num_layers": 2,
    "tft_graph_temporal_evolution": False, "tft_graph_edge_features": False,
    "tft_vsn_low_rank_threshold": 64, "model_id": "compare_tft",
}

def make_cfg(**overrides):
    d = dict(SHARED); d.update(overrides)
    class Cfg: pass
    c = Cfg()
    for k, v in d.items(): setattr(c, k, v)
    return c

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def run_model(label, model, cfg):
    from data_provider.data_factory import data_provider
    set_seed(SEED)

    _, train_loader = data_provider(cfg, "train")
    _, val_loader   = data_provider(cfg, "val")
    _, test_loader  = data_provider(cfg, "test")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=False)
    criterion = torch.nn.MSELoss()

    best_val, best_state, patience_cnt = float("inf"), None, 0
    train_hist, val_hist = [], []
    t0 = time.time()

    for epoch in range(1, cfg.train_epochs + 1):
        model.train(); ep_loss = 0.0; nb = 0
        for bx, by, bxm, bym in train_loader:
            bx  = bx.float().to(DEVICE); by  = by.float().to(DEVICE)
            bxm = bxm.float().to(DEVICE); bym = bym.float().to(DEVICE)
            dec_inp = torch.zeros_like(by[:, -cfg.pred_len:, :]).to(DEVICE)
            dec_inp = torch.cat([by[:, :cfg.label_len, :], dec_inp], dim=1)
            out = model(bx, bxm, dec_inp, bym)
            out = out[:, -cfg.pred_len:, :]
            loss = criterion(out, by[:, -cfg.pred_len:, :])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item(); nb += 1

        train_loss = ep_loss / max(nb, 1); train_hist.append(train_loss)

        model.eval(); vl = 0.0; nv = 0
        with torch.no_grad():
            for bx, by, bxm, bym in val_loader:
                bx  = bx.float().to(DEVICE); by  = by.float().to(DEVICE)
                bxm = bxm.float().to(DEVICE); bym = bym.float().to(DEVICE)
                dec_inp = torch.zeros_like(by[:, -cfg.pred_len:, :]).to(DEVICE)
                dec_inp = torch.cat([by[:, :cfg.label_len, :], dec_inp], dim=1)
                out = model(bx, bxm, dec_inp, bym)
                vl += criterion(out[:, -cfg.pred_len:, :], by[:, -cfg.pred_len:, :]).item(); nv += 1
        val_loss = vl / max(nv, 1); val_hist.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{label}] Ep {epoch:3d}/{cfg.train_epochs} | "
                  f"train={train_loss:.4f} val={val_loss:.4f} best={best_val:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if patience_cnt >= cfg.patience:
            print(f"  [{label}] Early stop ep {epoch}")
            break

    if best_state: model.load_state_dict(best_state)
    model.eval(); tl = 0.0; nt = 0
    with torch.no_grad():
        for bx, by, bxm, bym in test_loader:
            bx  = bx.float().to(DEVICE); by  = by.float().to(DEVICE)
            bxm = bxm.float().to(DEVICE); bym = bym.float().to(DEVICE)
            dec_inp = torch.zeros_like(by[:, -cfg.pred_len:, :]).to(DEVICE)
            dec_inp = torch.cat([by[:, :cfg.label_len, :], dec_inp], dim=1)
            out = model(bx, bxm, dec_inp, bym)
            tl += criterion(out[:, -cfg.pred_len:, :], by[:, -cfg.pred_len:, :]).item(); nt += 1
    test_loss = tl / max(nt, 1)
    elapsed = time.time() - t0
    gap = (best_val - train_hist[-1]) / train_hist[-1] * 100

    print(f"\n  [{label}] FINAL: train={train_hist[-1]:.4f} val={best_val:.4f} test={test_loss:.4f} gap={gap:.1f}% params={count_params(model):,} time={elapsed:.0f}s")
    return {"label": label, "train": train_hist[-1], "val": best_val, "test": test_loss, "gap": gap, "params": count_params(model), "time": elapsed}


# ── Run Custom TFT ───────────────────────────────────────────────────
print("="*70)
print("EXPERIMENT: CustomTFT vs NixtlaTFT — identical training loop")
print("Model: d=64, 2 layers, dropout=0.3, RevIN=True, backbone=LSTM")
print("Data: ETTh1, seq=96, pred=96, batch=256, epochs=30, patience=7")
print("="*70)

from models.TemporalFusionTransformer import Model as CustomTFT
cfg_custom = make_cfg(model="TemporalFusionTransformer")
set_seed(SEED)
m_custom = CustomTFT(cfg_custom).to(DEVICE)
print(f"\nCustom TFT params: {count_params(m_custom):,}")
r_custom = run_model("CustomTFT", m_custom, cfg_custom)

# ── Run Nixtla TFT ───────────────────────────────────────────────────
try:
    from models.TFT_Nixtla import Model as NixtlaTFT
    cfg_nixtla = make_cfg(model="TFT_Nixtla")
    set_seed(SEED)
    m_nixtla = NixtlaTFT(cfg_nixtla).to(DEVICE)
    print(f"\nNixtla TFT params: {count_params(m_nixtla):,}")
    r_nixtla = run_model("NixtlaTFT", m_nixtla, cfg_nixtla)
    results = [r_custom, r_nixtla]
except Exception as e:
    print(f"\nNixtla TFT failed: {e}")
    results = [r_custom]

print("\n" + "="*70)
print(f"  {'Model':<14} {'Train':>8} {'Val':>8} {'Test':>8} {'Gap':>8} {'Params':>10}")
print("  " + "-"*60)
for r in results:
    print(f"  {r['label']:<14} {r['train']:>8.4f} {r['val']:>8.4f} {r['test']:>8.4f} {r['gap']:>7.1f}% {r['params']:>10,}")
print("="*70)
