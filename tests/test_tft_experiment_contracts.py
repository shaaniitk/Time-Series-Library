import numpy as np
import torch
import pytest
from types import SimpleNamespace
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from models.TemporalFusionTransformer import Model
from utils.tft_schema import inverse_transform_selected, resolve_target_positions, select_tft_truth
from utils.metrics import quantile_metric


def test_nonlast_ms_target_mapping_end_to_end():
    args = SimpleNamespace(model="TemporalFusionTransformer", enc_in=3, c_out=1, tft_target_pos=[1])
    batch_y = torch.tensor(
        [
            [[0.0, 10.0, 20.0], [1.0, 11.0, 21.0], [2.0, 12.0, 22.0]],
            [[3.0, 13.0, 23.0], [4.0, 14.0, 24.0], [5.0, 15.0, 25.0]],
        ]
    )
    target_pos = resolve_target_positions(args)
    selected = select_tft_truth(batch_y, pred_len=2, target_positions=target_pos)
    expected = torch.tensor([[[11.0], [12.0]], [[14.0], [15.0]]])
    assert torch.equal(selected, expected)


def test_noncontiguous_multioutput_target_mapping():
    args = SimpleNamespace(model="TemporalFusionTransformer", enc_in=5, c_out=2, tft_target_pos=[4, 1])
    batch_y = torch.arange(2 * 4 * 5, dtype=torch.float32).view(2, 4, 5)
    selected = select_tft_truth(batch_y, pred_len=2, target_positions=resolve_target_positions(args))
    expected = torch.stack([batch_y[:, -2:, 4], batch_y[:, -2:, 1]], dim=-1)
    assert torch.equal(selected, expected)


def test_selected_target_inverse_transform():
    scaler = SimpleNamespace(
        mean_=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        scale_=np.array([10.0, 20.0, 30.0], dtype=np.float64),
    )
    values = np.array([[[1.0, -1.0]]], dtype=np.float64)
    restored = inverse_transform_selected(values, scaler, target_positions=(2, 0))
    expected = np.array([[[330.0, 90.0]]], dtype=np.float64)
    np.testing.assert_allclose(restored, expected)


def test_tft_path_never_uses_f_dim():
    args = SimpleNamespace(model="TemporalFusionTransformer", enc_in=4, c_out=1, tft_target_pos=[1])
    batch_y = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]]])
    selected = select_tft_truth(batch_y, pred_len=1, target_positions=resolve_target_positions(args))
    assert selected.item() == 11.0


def _build_tft_args():
    return SimpleNamespace(
        task_name="long_term_forecast",
        model="TemporalFusionTransformer",
        tft_declared_regular_sampling=True,
        data="custom_tft_exp_contracts",
        seq_len=12,
        label_len=6,
        pred_len=3,
        enc_in=4,
        dec_in=2,
        c_out=2,
        d_model=16,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        use_gpu=False,
        use_multi_gpu=False,
        learning_rate=1e-3,
        features="M",
        tft_observed_pos=[0, 1, 2, 3],
        tft_static_pos=[],
        tft_target_pos=[2, 0],
        tft_allow_custom_known=True,
        tft_known_len=5,
        tft_known_max_channels=16,
        tft_known_feature_names=[f"known_{i}" for i in range(5)],
        tft_use_quantile_head=True,
        tft_output_quantiles=[0.1, 0.5, 0.9],
        tft_output_mode="joint",
        tft_use_revin=True,
        tft_revin_affine=True,
        tft_use_regime_moe=True,
        tft_num_regimes=3,
        tft_num_moe_experts=4,
        tft_moe_top_k=2,
        tft_moe_hidden_size=16,
        tft_moe_noise_epsilon=1e-2,
        tft_moe_aux_loss_coeff=0.05,
        tft_temporal_backbone="lstm",
        tft_temporal_backbone_layers=1,
        tft_temporal_kernel_size=3,
        tft_temporal_hidden_size=16,
        tft_use_swiglu=False,
        tft_full_attention=False,
        tft_cross_variable_mixing=False,
        tft_vsn_residual_bypass=True,
        tft_dual_attention_fusion=False,
        tft_use_explicit_cross_attention=False,
        tft_attention_position_bias="none",
        tft_attention_backend="exact",
        tft_lag_scales=[1, 2],
        tft_use_lag_attention=False,
        tft_use_higher_order=False,
        tft_interaction_order=2,
        tft_interaction_rank=None,
        tft_payload_stack_layers=True,
        tft_vsn_n_selection_heads=1,
        tft_vsn_per_feature_gating=False,
        tft_covariate_reattention=False,
        tft_graph_type="dense",
        tft_graph_top_k=5,
        tft_graph_num_layers=2,
        tft_graph_temporal_evolution=False,
        tft_graph_edge_features=False,
        tft_vsn_low_rank_threshold=64,
        tft_per_target_heads=False,
        tft_mlp_quantile_projection=False,
        tft_quantile_projection_ff_size=0,
        tft_moe_capacity_factor=1.25,
        num_workers=0,
        checkpoints="./checkpoints",
        patience=2,
        train_epochs=1,
        lradj="type1",
        loss="MSE",
        use_amp=False,
        inverse=False,
    )


def test_structured_output_shapes():
    args = _build_tft_args()
    model = Model(args)
    x_enc = torch.randn(2, args.seq_len, args.enc_in)
    x_mark_enc = torch.randn(2, args.seq_len, args.tft_known_len)
    x_dec = torch.randn(2, args.label_len + args.pred_len, args.c_out)
    x_mark_dec = torch.randn(2, args.label_len + args.pred_len, args.tft_known_len)
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_auxiliary=True)
    assert tuple(output.point_forecast.shape) == (2, args.pred_len, args.c_out)
    assert tuple(output.point_full.shape) == (2, args.seq_len + args.pred_len, args.c_out)
    assert tuple(output.quantile_forecast.shape) == (2, args.pred_len, 3, args.c_out)
    # Semantics-v2 MoE starts as an exact-neutral residual. Routing is observed
    # internally, but it cannot leak an auxiliary objective before the branch
    # strength moves away from zero.
    assert output.moe_importance_sum is None
    assert output.moe_load_sum is None
    assert output.moe_token_count is None
    assert output.moe_aux_loss.item() == 0.0


def test_global_moe_reduction_uses_counts():
    exp = Exp_Long_Term_Forecast(_build_tft_args())
    model_output = SimpleNamespace(
        point_full=torch.zeros(2, 15, 2),
        quantile_forecast=None,
        moe_importance_sum=torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
        moe_load_sum=torch.tensor([[3.0, 1.0], [1.0, 3.0]], dtype=torch.float32),
        moe_token_count=torch.tensor([3.0, 5.0], dtype=torch.float32),
    )
    _, _, aux_loss = exp._extract_outputs_and_aux(model_output)
    expected = (
        exp._cv_squared(torch.tensor([2.0, 2.0], dtype=torch.float32))
        + exp._cv_squared(torch.tensor([4.0, 4.0], dtype=torch.float32))
    )
    assert torch.allclose(aux_loss, expected)


def test_point_mode_has_no_untrained_quantile_head():
    args = _build_tft_args()
    args.tft_use_quantile_head = False
    args.tft_output_mode = "point"
    model = Model(args)
    x_enc = torch.randn(1, args.seq_len, args.enc_in)
    x_mark_enc = torch.randn(1, args.seq_len, args.tft_known_len)
    x_dec = torch.randn(1, args.label_len + args.pred_len, args.c_out)
    x_mark_dec = torch.randn(1, args.label_len + args.pred_len, args.tft_known_len)
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_auxiliary=True)
    assert output.quantile_forecast is None


def test_unsorted_quantiles_canonicalized_once():
    args = _build_tft_args()
    args.tft_output_quantiles = [0.9, 0.1, 0.5]
    model = Model(args)
    assert model.quantiles == [0.1, 0.5, 0.9]


def test_quantile_only_evaluates_trained_output():
    args = _build_tft_args()
    args.tft_output_mode = "quantile"
    model = Model(args)
    assert model.temporal_fusion_decoder.out_projection is None
    with torch.no_grad():
        if hasattr(model.quantile_projection, "weight"):
            model.quantile_projection.weight.zero_()
            model.quantile_projection.bias.copy_(torch.tensor([-3.0, 0.0, 2.0] * args.c_out))
    x_enc = torch.randn(1, args.seq_len, args.enc_in)
    x_mark_enc = torch.randn(1, args.seq_len, args.tft_known_len)
    x_dec = torch.randn(1, args.label_len + args.pred_len, args.c_out)
    x_mark_dec = torch.randn(1, args.label_len + args.pred_len, args.tft_known_len)
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_auxiliary=True)
    median = output.quantile_forecast[:, :, 1, :]
    assert torch.allclose(output.point_forecast, median)


def test_quantile_outputs_do_not_cross():
    args = _build_tft_args()
    model = Model(args)
    x_enc = torch.randn(2, args.seq_len, args.enc_in)
    x_mark_enc = torch.randn(2, args.seq_len, args.tft_known_len)
    x_dec = torch.randn(2, args.label_len + args.pred_len, args.c_out)
    x_mark_dec = torch.randn(2, args.label_len + args.pred_len, args.tft_known_len)
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_auxiliary=True)
    qp = output.quantile_forecast
    assert torch.all(qp[:, :, 1:, :] >= qp[:, :, :-1, :])


def test_revin_effective_scale_is_positive():
    args = _build_tft_args()
    model = Model(args)
    assert torch.all(model.revin.affine_weight > 0)


def test_quantile_order_survives_revin_denormalization():
    args = _build_tft_args()
    args.tft_output_mode = "quantile"
    model = Model(args)
    x_enc = torch.randn(1, args.seq_len, args.enc_in)
    x_mark_enc = torch.randn(1, args.seq_len, args.tft_known_len)
    x_dec = torch.randn(1, args.label_len + args.pred_len, args.c_out)
    x_mark_dec = torch.randn(1, args.label_len + args.pred_len, args.tft_known_len)
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_auxiliary=True)
    qp = output.quantile_forecast
    assert torch.all(qp[:, :, 1:, :] >= qp[:, :, :-1, :])


def test_joint_mode_rejects_nonpositive_coeffs_early():
    args = _build_tft_args()
    args.tft_point_loss_coeff = 0.0
    with pytest.raises(ValueError, match="requires positive"):
        Exp_Long_Term_Forecast(args)


def test_quantile_metric_reports_calibration_stats():
    quantiles = (0.1, 0.5, 0.9)
    pred_quantiles = np.array(
        [
            [
                [[0.0], [1.0], [2.0]],
                [[1.0], [2.0], [3.0]],
            ]
        ],
        dtype=np.float64,
    )
    true = np.array([[[1.5], [2.5]]], dtype=np.float64)
    summary = quantile_metric(pred_quantiles, true, quantiles)
    assert set(summary) == {"pinball", "coverage", "interval_width", "crossing_rate"}
    assert summary["pinball"] > 0.0
    assert summary["coverage"] == 1.0
    assert summary["interval_width"] == 2.0
    assert summary["crossing_rate"] == 0.0
