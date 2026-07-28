#!/usr/bin/env python
"""RevIN ablation: 4 epochs, d=32, to confirm distribution shift is the problem."""
import os, sys, time, random
os.environ["MIOPEN_LOG_LEVEL"] = "3"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import torch, numpy as np
sys.path.insert(0, "/home/kalki/Documents/workspace/Time-Series-Library")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

BASE = dict(
    task_name="long_term_forecast", data="ETTh1", root_path="./dataset/ETT-small/",
    data_path="ETTh1.csv", features="M", target="OT", freq="h",
    seq_len=96, label_len=48, pred_len=96, enc_in=7, dec_in=7, c_out=7,
    embed="timeF", seasonal_patterns="Monthly", inverse=False,
    num_workers=0, augmentation_ratio=0,
    d_model=32, n_heads=4, e_layers=1, d_layers=1, d_ff=64,
    dropout=0.3, learning_rate=3e-4, train_epochs=6, batch_size=256, patience=4,
    channel_independence=1, decomp_method="moving_avg", use_norm=1,
    down_sampling_layers=0, down_sampling_window=1, down_sampling_method=None,
    seg_len=96, top_k=5, num_kernels=6, expand=2, d_conv=4, patch_len=16,
    node_dim=10, gcn_depth=2, gcn_dropout=0.3, propalpha=0.3,
    conv_channel=32, skip_channel=32, individual=False, moving_avg=25, factor=1, distil=True,
    itr=1, des="test", loss="MSE", lradj="type1", use_amp=False,
    use_gpu=True, gpu=0, gpu_type="cuda", use_multi_gpu=False, devices="0",
    checkpoints="./checkpoints/", mask_rate=0.25, anomaly_ratio=0.25,
    p_hidden_dims=[128, 128], p_hidden_layers=2, use_dtw=False,
    tft_observed_pos=None, tft_static_pos=None, tft_target_pos=None,
    tft_use_swiglu=False, tft_full_attention=False, tft_cross_variable_mixing=False,
    tft_allow_custom_known=False, tft_vsn_residual_bypass=True,
    tft_dual_attention_fusion=False, tft_use_lag_attention=False, tft_lag_scales="1,2,4",
    tft_temporal_backbone="lstm", tft_temporal_backbone_layers=1,
    tft_temporal_kernel_size=3, tft_temporal_hidden_size=0,
    tft_use_higher_order=False, tft_interaction_order=2, tft_interaction_rank=0,
    tft_use_regime_moe=False, tft_use_explicit_cross_attention=False,
    tft_cross_attention_type="full", tft_attention_position_bias="none",
    tft_attention_backend="exact", tft_rope_base=10000.0, tft_alibi_scale=1.0,
    tft_revin_affine=True, tft_use_quantile_head=False, tft_output_quantiles="0.1,0.5,0.9",
    tft_num_regimes=4, tft_num_moe_experts=4, tft_moe_top_k=2, tft_moe_hidden_size=0,
    tft_moe_noise_epsilon=1e-2, tft_moe_aux_loss_coeff=0.0, tft_moe_capacity_factor=1.25,
    tft_per_target_heads=False, tft_vsn_per_feature_gating=False,
    tft_covariate_reattention=False, tft_graph_type="dense", tft_graph_top_k=5,
    tft_graph_num_layers=2, tft_graph_temporal_evolution=False, tft_graph_edge_features=False,
    tft_vsn_low_rank_threshold=64, model_id="revin_ablation",
)

def make_cfg(**kw):
    d = dict(BASE); d.update(kw)
    class C: pass
    c = C()
    [setattr(c, k, v) for k, v in d.items()]
    return c

def run(label, cfg):
    from models.TemporalFusionTransformer import Model
    from data_provider.data_factory import data_provider
    set_seed(42)
    _, tl = data_provider(cfg, "train")
    _, vl = data_provider(cfg, "val")
    _, xl = data_provider(cfg, "test")
    model = Model(cfg).to(DEVICE)
    print(f"  [{label}] params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    crit = torch.nn.MSELoss()
    best_val, best_state, pat = float("inf"), None, 0
    for ep in range(1, cfg.train_epochs + 1):
        model.train(); el = 0; nb = 0
        for bx, by, bxm, bym in tl:
            bx = bx.float().to(DEVICE); by = by.float().to(DEVICE)
            bxm = bxm.float().to(DEVICE); bym = bym.float().to(DEVICE)
            di = torch.cat([by[:, :cfg.label_len, :], torch.zeros_like(by[:, -cfg.pred_len:, :]).to(DEVICE)], dim=1)
            out = model(bx, bxm, di, bym)
            loss = crit(out[:, -cfg.pred_len:, :], by[:, -cfg.pred_len:, :])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); el += loss.item(); nb += 1
        tl_ = el / nb
        model.eval(); vv = 0; nv = 0
        with torch.no_grad():
            for bx, by, bxm, bym in vl:
                bx = bx.float().to(DEVICE); by = by.float().to(DEVICE)
                bxm = bxm.float().to(DEVICE); bym = bym.float().to(DEVICE)
                di = torch.cat([by[:, :cfg.label_len, :], torch.zeros_like(by[:, -cfg.pred_len:, :]).to(DEVICE)], dim=1)
                out = model(bx, bxm, di, bym)
                vv += crit(out[:, -cfg.pred_len:, :], by[:, -cfg.pred_len:, :]).item(); nv += 1
        vl_ = vv / nv
        if vl_ < best_val: best_val = vl_; best_state = {k: v.clone() for k, v in model.state_dict().items()}; pat = 0
        else: pat += 1
        print(f"  [{label}] ep{ep}: train={tl_:.4f}  val={vl_:.4f}  gap={100*(vl_-tl_)/tl_:.0f}%")
        if pat >= cfg.patience: print(f"  [{label}] early stop"); break
    if best_state: model.load_state_dict(best_state)
    model.eval(); tt = 0; nt = 0
    with torch.no_grad():
        for bx, by, bxm, bym in xl:
            bx = bx.float().to(DEVICE); by = by.float().to(DEVICE)
            bxm = bxm.float().to(DEVICE); bym = bym.float().to(DEVICE)
            di = torch.cat([by[:, :cfg.label_len, :], torch.zeros_like(by[:, -cfg.pred_len:, :]).to(DEVICE)], dim=1)
            out = model(bx, bxm, di, bym)
            tt += crit(out[:, -cfg.pred_len:, :], by[:, -cfg.pred_len:, :]).item(); nt += 1
    return best_val, tt / nt, tl_

print("=" * 60)
print("RevIN ablation — d=32, 1 layer, 6 epochs, batch=256")
print("=" * 60)

print("\nRun 1: RevIN=OFF")
v1, t1, tr1 = run("RevIN=OFF", make_cfg(tft_use_revin=False))

print("\nRun 2: RevIN=ON (instance norm per sample)")
v2, t2, tr2 = run("RevIN=ON",  make_cfg(tft_use_revin=True))

print("\n" + "=" * 60)
print(f"  {'':12} {'Train':>8} {'Val':>8} {'Test':>8} {'Gap':>8}")
print(f"  {'RevIN=OFF':12} {tr1:>8.4f} {v1:>8.4f} {t1:>8.4f} {100*(v1-tr1)/tr1:>7.0f}%")
print(f"  {'RevIN=ON':12} {tr2:>8.4f} {v2:>8.4f} {t2:>8.4f} {100*(v2-tr2)/tr2:>7.0f}%")
print(f"  Val improvement from RevIN: {100*(v1-v2)/v1:.1f}%")
print("=" * 60)
