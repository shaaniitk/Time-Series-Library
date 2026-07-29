import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import TemporalFusionTransformer as tsl_tft
from utils.tft_synthetic import make_multiscale_tft_tensors


def _build_cfg(use_revin: bool):
    return SimpleNamespace(
        task_name="long_term_forecast",
        data="custom_revin_ablation",
        seq_len=12,
        label_len=6,
        pred_len=2,
        enc_in=4,
        dec_in=2,
        c_out=2,
        d_model=16,
        n_heads=2,
        dropout=0.1,
        embed="timeF",
        freq="h",
        e_layers=1,
        tft_use_swiglu=False,
        tft_full_attention=False,
        tft_dual_attention_fusion=False,
        tft_use_explicit_cross_attention=False,
        tft_cross_attention_type="full",
        tft_attention_position_bias="none",
        tft_attention_backend="exact",
        tft_rope_base=10000.0,
        tft_alibi_scale=1.0,
        tft_use_revin=use_revin,
        tft_revin_affine=True,
        tft_use_quantile_head=False,
        tft_output_quantiles=[0.1, 0.5, 0.9],
        tft_use_lag_attention=False,
        tft_lag_scales=[1, 2, 4],
        tft_temporal_backbone="lstm",
        tft_temporal_backbone_layers=1,
        tft_temporal_kernel_size=3,
        tft_temporal_hidden_size=16,
        tft_use_higher_order=False,
        tft_interaction_order=2,
        tft_interaction_rank=8,
        tft_use_regime_moe=False,
        tft_num_regimes=3,
        tft_num_moe_experts=2,
        tft_moe_top_k=2,
        tft_moe_hidden_size=24,
        tft_moe_noise_epsilon=1e-2,
        tft_moe_aux_loss_coeff=0.0,
        tft_payload_stack_layers=True,
        tft_cross_variable_mixing=False,
        tft_vsn_residual_bypass=True,
        tft_vsn_n_selection_heads=1,
        tft_per_target_heads=False,
        tft_vsn_per_feature_gating=False,
        tft_covariate_reattention=False,
        tft_moe_capacity_factor=1.25,
        tft_vsn_low_rank_threshold=64,
        tft_graph_type="dense",
        tft_graph_top_k=5,
        tft_graph_num_layers=2,
        tft_graph_temporal_evolution=False,
        tft_graph_edge_features=False,
        tft_allow_custom_known=True,
        tft_known_len=4,
        tft_known_max_channels=16,
        tft_known_feature_names=[f"known_{i}" for i in range(4)],
        tft_observed_pos=list(range(4)),
        tft_static_pos=[],
        tft_target_pos=[0, 1],
    )


def _make_single_batch(cfg):
    torch.manual_seed(42)
    tensors = make_multiscale_tft_tensors(
        seq_len=cfg.seq_len,
        label_len=cfg.label_len,
        pred_len=cfg.pred_len,
        enc_in=cfg.enc_in,
        c_out=cfg.c_out,
        known_len=cfg.tft_known_len,
        n_samples=2,
        noise_std=0.01,
    )
    return (
        tensors["x_enc"],
        tensors["x_mark_enc"],
        tensors["x_dec"],
        tensors["x_mark_dec"],
        tensors["y_future"],
    )


def _run_brief_training(cfg, steps=2):
    torch.manual_seed(42)
    model = tsl_tft.Model(cfg).float()
    x_enc, x_mark_enc, x_dec, x_mark_dec, y = _make_single_batch(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = torch.nn.MSELoss()
    losses = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred = out[:, -cfg.pred_len:, :]
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses


def test_revin_forward_backward_smoke():
    for use_revin in (False, True):
        cfg = _build_cfg(use_revin=use_revin)
        model, losses = _run_brief_training(cfg)
        assert torch.isfinite(torch.tensor(losses)).all()
        assert len(losses) == 2
        if use_revin:
            assert model.revin is not None
            assert torch.all(model.revin.affine_weight > 0)


def test_revin_is_not_worse_than_manual_norm_on_synthetic_smoke():
    _, off_losses = _run_brief_training(_build_cfg(use_revin=False))
    _, on_losses = _run_brief_training(_build_cfg(use_revin=True))
    assert on_losses[-1] <= off_losses[-1] * 1.10, (
        f"RevIN synthetic smoke regressed too much: off={off_losses[-1]:.4f}, on={on_losses[-1]:.4f}"
    )
