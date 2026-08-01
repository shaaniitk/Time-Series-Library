from types import SimpleNamespace
import os
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from run import build_parser, normalize_args
from utils.tft_config import apply_tft_profile
from models.TemporalFusionTransformer import Model


def _direct_args(**overrides):
    base = dict(
        task_name="long_term_forecast",
        is_training=1,
        model_id="tft-profile-test",
        model="TemporalFusionTransformer",
        tft_declared_regular_sampling=True,
        data="ETTh1",
        root_path="./data/ETT/",
        data_path="ETTh1.csv",
        features="M",
        target="OT",
        freq="h",
        checkpoints="./checkpoints/",
        seq_len=24,
        label_len=12,
        pred_len=4,
        seasonal_patterns="Monthly",
        inverse=False,
        mask_rate=0.25,
        anomaly_ratio=0.25,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=24,
        num_workers=0,
        itr=1,
        train_epochs=1,
        batch_size=2,
        patience=1,
        learning_rate=1e-3,
        des="test",
        loss="MSE",
        lradj="type1",
        use_amp=False,
        use_gpu=False,
        gpu=0,
        gpu_type="cuda",
        use_multi_gpu=False,
        devices="0",
        p_hidden_dims=[8, 8],
        p_hidden_layers=2,
        use_dtw=False,
        augmentation_ratio=0,
        seed=2,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",
        patch_len=16,
        node_dim=10,
        gcn_depth=2,
        gcn_dropout=0.3,
        propalpha=0.3,
        conv_channel=32,
        skip_channel=32,
        individual=False,
        tft_profile="extended_safe",
        tft_observed_pos=None,
        tft_static_pos=None,
        tft_target_pos=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_cli_and_model_defaults_match():
    parser = build_parser()
    cli_args = normalize_args(parser.parse_args([
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", "tft-profile-test",
        "--model", "TemporalFusionTransformer",
        "--data", "ETTh1",
        "--no_use_gpu",
    ]))
    direct_args = apply_tft_profile(_direct_args())

    fields = [
        "tft_profile",
        "tft_extension_semantics_version",
        "tft_digest_schema",
        "tft_temporal_backbone",
        "tft_use_quantile_head",
        "tft_output_mode",
        "tft_full_attention",
        "tft_cross_variable_mixing",
        "tft_vsn_residual_bypass",
        "tft_vsn_per_feature_gating",
        "tft_use_fft_branch",
        "tft_graph_type",
        "tft_config_digest",
    ]
    for field in fields:
        assert getattr(cli_args, field) == getattr(direct_args, field), field


def test_canonical_profile_flags():
    args = apply_tft_profile(_direct_args(tft_profile="canonical"))
    assert args.tft_profile == "canonical"
    assert args.tft_temporal_backbone == "lstm"
    assert args.tft_use_quantile_head is True
    assert args.tft_output_mode == "joint"
    assert args.tft_full_attention is False
    assert args.tft_vsn_per_feature_gating is False
    assert args.tft_use_fft_branch is False
    assert args.tft_use_regime_moe is False


def test_profile_rejects_incompatible_override():
    with pytest.raises(ValueError, match="canonical"):
        apply_tft_profile(_direct_args(tft_profile="canonical", tft_full_attention=True))


def test_config_digest_changes_for_material_flag():
    base = apply_tft_profile(_direct_args(tft_profile="extended_safe"))
    changed = apply_tft_profile(_direct_args(tft_profile="extended_safe", tft_use_fft_branch=True))
    assert base.tft_config_digest != changed.tft_config_digest


def test_tft_warns_that_d_ff_is_ignored_and_digest_is_stable():
    with pytest.warns(UserWarning, match="ignores d_ff"):
        changed = apply_tft_profile(_direct_args(tft_profile="extended_safe", d_ff=32))
    base = apply_tft_profile(_direct_args(tft_profile="extended_safe", d_ff=2048))
    assert changed.tft_config_digest == base.tft_config_digest
    assert changed.tft_temporal_backbone_layers_scope == "gated_tcn_and_hybrid_tcn_lstm_only"


def test_identical_resolved_config_reuses_digest():
    left = apply_tft_profile(_direct_args(tft_profile="experimental_full", tft_use_fft_branch=True))
    right = apply_tft_profile(_direct_args(tft_profile="experimental_full", tft_use_fft_branch=True))
    assert left.tft_config_digest == right.tft_config_digest


def test_model_import_does_not_mutate_env_vars():
    code = (
        "import os; "
        "os.environ.pop('MIOPEN_LOG_LEVEL', None); "
        "os.environ.pop('HSA_OVERRIDE_GFX_VERSION', None); "
        "import models.TemporalFusionTransformer; "
        "print(os.environ.get('MIOPEN_LOG_LEVEL')); "
        "print(os.environ.get('HSA_OVERRIDE_GFX_VERSION'))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(__file__))
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip().splitlines() == ["None", "None"]


def _canonical_static_args():
    return apply_tft_profile(_direct_args(
        tft_profile="canonical",
        data="custom_tft_a03",
        seq_len=4,
        label_len=2,
        pred_len=2,
        enc_in=3,
        dec_in=1,
        c_out=1,
        dropout=0.0,
        tft_observed_pos=[0, 1],
        tft_static_pos=[2],
        tft_target_pos=[0],
    ))    


def test_debug_on_off_match_for_finite_eval_inputs():
    args_off = apply_tft_profile(_direct_args(
        tft_profile="extended_safe",
        data="custom_tft_a08",
        seq_len=8,
        label_len=4,
        pred_len=2,
        enc_in=3,
        dec_in=1,
        c_out=1,
        dropout=0.0,
        tft_known_len=4,
        tft_known_feature_names=[f"known_{i}" for i in range(4)],
        tft_allow_custom_known=True,
        tft_observed_pos=[0, 1],
        tft_static_pos=[2],
        tft_target_pos=[0],
        tft_debug_checks=False,
    ))
    args_on = apply_tft_profile(_direct_args(
        tft_profile="extended_safe",
        data="custom_tft_a08",
        seq_len=8,
        label_len=4,
        pred_len=2,
        enc_in=3,
        dec_in=1,
        c_out=1,
        dropout=0.0,
        tft_known_len=4,
        tft_known_feature_names=[f"known_{i}" for i in range(4)],
        tft_allow_custom_known=True,
        tft_observed_pos=[0, 1],
        tft_static_pos=[2],
        tft_target_pos=[0],
        tft_debug_checks=True,
    ))
    model_off = Model(args_off).eval()
    model_on = Model(args_on).eval()
    model_on.load_state_dict(model_off.state_dict())
    x_enc, x_mark_enc, x_dec, x_mark_dec = _make_io(args_off, batch_size=2)
    with torch.no_grad():
        out_off = model_off(x_enc, x_mark_enc, x_dec, x_mark_dec, return_auxiliary=True)
        out_on = model_on(x_enc, x_mark_enc, x_dec, x_mark_dec, return_auxiliary=True)
    assert torch.allclose(out_off.point_forecast, out_on.point_forecast)
    if out_off.quantile_forecast is not None:
        assert torch.allclose(out_off.quantile_forecast, out_on.quantile_forecast)


def test_invalid_shape_still_fails_with_debug_off():
    args = apply_tft_profile(_direct_args(
        tft_profile="extended_safe",
        data="custom_tft_a08",
        seq_len=8,
        label_len=4,
        pred_len=2,
        enc_in=3,
        dec_in=1,
        c_out=1,
        dropout=0.0,
        tft_known_len=4,
        tft_known_feature_names=[f"known_{i}" for i in range(4)],
        tft_allow_custom_known=True,
        tft_observed_pos=[0, 1],
        tft_static_pos=[2],
        tft_target_pos=[0],
        tft_debug_checks=False,
    ))
    model = Model(args)
    x_enc, x_mark_enc, x_dec, x_mark_dec = _make_io(args, batch_size=1)
    with pytest.raises(ValueError):
        model(x_enc, x_mark_enc[:, :, :2], x_dec, x_mark_dec)


def test_canonical_pointwise_embedding_does_not_mix_neighboring_timesteps():
    args = _canonical_static_args()
    model = Model(args).eval()
    emb = model.embedding.observed_embedding[0]
    emb.projection.weight.data.fill_(1.0)

    x_enc = torch.zeros(1, args.seq_len, args.enc_in)
    x_enc[0, 1, 0] = 3.0
    x_mark_enc = torch.zeros(1, args.seq_len, 4)
    x_dec = torch.zeros(1, args.label_len + args.pred_len, args.c_out)
    x_mark_dec = torch.zeros(1, args.label_len + args.pred_len, 4)
    static_values = torch.zeros(1, 1)

    _, observed_input, _ = model.embedding(
        x_enc, x_mark_enc, x_dec, x_mark_dec, static_values=static_values
    )
    first_feature = observed_input[0, :, 0, 0]
    assert torch.equal(first_feature, torch.tensor([0.0, 3.0, 0.0, 0.0]))


def test_canonical_profile_uses_single_static_vsn_and_no_position_buffers():
    args = _canonical_static_args()
    model = Model(args)
    assert model.static_encoder.canonical_mode is True
    assert model.static_encoder.static_vsn is not None
    assert model.static_encoder.static_vsn_cs is None
    assert model.static_encoder.static_vsn_cc is None
    assert model.static_encoder.static_vsn_ch is None
    assert model.static_encoder.static_vsn_ce is None
    static_vsn_count = sum(
        1 for module in model.static_encoder.modules()
        if module.__class__.__name__ == "VariableSelectionNetwork"
    )
    assert static_vsn_count == 1
    position_buffers = [name for name, _ in model.embedding.named_buffers() if "position_embedding" in name]
    assert position_buffers == []


def test_canonical_profile_reduces_embedding_and_static_encoder_parameters():
    canonical = Model(_canonical_static_args())
    extended = Model(apply_tft_profile(_direct_args(
        tft_profile="extended_safe",
        data="custom_tft_a03",
        seq_len=4,
        label_len=2,
        pred_len=2,
        enc_in=3,
        dec_in=1,
        c_out=1,
        dropout=0.0,
        tft_observed_pos=[0, 1],
        tft_static_pos=[2],
        tft_target_pos=[0],
    )))

    def count_named(prefix, model):
        return sum(p.numel() for name, p in model.named_parameters() if name.startswith(prefix))

    canonical_scope = count_named("embedding", canonical) + count_named("static_encoder", canonical)
    extended_scope = count_named("embedding", extended) + count_named("static_encoder", extended)
    assert canonical_scope < extended_scope


def _make_io(args, batch_size=2):
    x_enc = torch.randn(batch_size, args.seq_len, args.enc_in)
    for static_idx in getattr(args, "tft_static_pos", []) or []:
        static_values = torch.randn(batch_size, 1)
        x_enc[:, :, static_idx] = static_values
    x_dec = torch.randn(batch_size, args.label_len + args.pred_len, args.c_out)
    known_len = len(getattr(args, "tft_known_feature_names", []) or []) or 4
    x_mark_enc = torch.randn(batch_size, args.seq_len, known_len)
    x_mark_dec = torch.randn(batch_size, args.label_len + args.pred_len, known_len)
    return x_enc, x_mark_enc, x_dec, x_mark_dec


def test_selected_profile_has_no_unexpected_dead_parameters():
    args = apply_tft_profile(_direct_args(
        tft_profile="canonical",
        data="custom_tft_a02",
        seq_len=8,
        label_len=4,
        pred_len=2,
        enc_in=3,
        dec_in=1,
        c_out=1,
        dropout=0.0,
        tft_known_len=4,
        tft_known_feature_names=[f"known_{i}" for i in range(4)],
        tft_allow_custom_known=True,
        tft_observed_pos=[0, 1],
        tft_static_pos=[2],
        tft_target_pos=[0],
    ))
    model = Model(args).train()
    x_enc, x_mark_enc, x_dec, x_mark_dec = _make_io(args, batch_size=2)
    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_auxiliary=True)
    loss = out.point_forecast.sum()
    if out.quantile_forecast is not None:
        loss = loss + out.quantile_forecast.sum()
    loss.backward()

    dead = sorted(
        name for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    )
    assert dead == []


def test_state_dict_keys_do_not_include_disabled_branches():
    args = apply_tft_profile(_direct_args(
        tft_profile="extended_safe",
        data="custom_tft_a02",
        seq_len=8,
        label_len=4,
        pred_len=2,
        enc_in=4,
        dec_in=2,
        c_out=2,
        dropout=0.0,
        tft_known_len=4,
        tft_known_feature_names=[f"known_{i}" for i in range(4)],
        tft_allow_custom_known=True,
        tft_observed_pos=[0, 1, 2, 3],
        tft_static_pos=[],
        tft_target_pos=[0, 1],
    ))
    model = Model(args)
    keys = set(model.state_dict().keys())
    forbidden_substrings = [
        "feature_gate_grn",
        "feature_gate_dropout",
        "residual_projection",
        "residual_gate",
        "static_encoder.static_vsn",
        "static_encoder.grns",
    ]
    assert not any(any(token in key for token in forbidden_substrings) for key in keys)


def test_parameter_count_regression_per_profile():
    canonical = Model(_canonical_static_args())
    extended = Model(apply_tft_profile(_direct_args(
        tft_profile="extended_safe",
        data="custom_tft_a02",
        seq_len=4,
        label_len=2,
        pred_len=2,
        enc_in=3,
        dec_in=1,
        c_out=1,
        dropout=0.0,
        tft_observed_pos=[0, 1],
        tft_static_pos=[2],
        tft_target_pos=[0],
    )))
    experimental = Model(apply_tft_profile(_direct_args(
        tft_profile="experimental_full",
        data="custom_tft_a02",
        seq_len=4,
        label_len=2,
        pred_len=2,
        enc_in=3,
        dec_in=1,
        c_out=1,
        dropout=0.0,
        tft_known_len=4,
        tft_known_feature_names=[f"known_{i}" for i in range(4)],
        tft_allow_custom_known=True,
        tft_observed_pos=[0, 1],
        tft_static_pos=[2],
        tft_target_pos=[0],
    )))
    count = lambda model: sum(p.numel() for p in model.parameters())
    assert count(canonical) < count(extended) < count(experimental)
