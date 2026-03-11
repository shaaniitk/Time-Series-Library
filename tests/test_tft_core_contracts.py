from types import SimpleNamespace
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.TemporalFusionTransformer import Model
from utils.tft_schema import resolve_tft_schema


def _build_args(task_name="long_term_forecast"):
    return SimpleNamespace(
        task_name=task_name,
        seq_len=24,
        label_len=12,
        pred_len=4,
        data="ETTh1",
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=16,
        dropout=0.1,
        embed="timeF",
        freq="h",
        n_heads=2,
        e_layers=1,
        tft_use_revin=False,
        tft_revin_affine=False,
        tft_use_quantile_head=False,
        tft_output_mode="point",
        tft_temporal_backbone="lstm",
        # These unit fixtures use a synthetic, gap-free row grid.
        tft_declared_regular_sampling=True,
    )


def test_short_term_rejected_before_m4_schema_lookup():
    args = _build_args(task_name="short_term_forecast")
    args.data = "m4"
    with pytest.raises(NotImplementedError, match="long_term_forecast only"):
        Model(args)


def test_unsupported_task_rejected_at_construction():
    args = _build_args(task_name="classification")
    with pytest.raises(NotImplementedError, match="supports long_term_forecast only"):
        Model(args)


def test_long_term_still_constructs():
    model = Model(_build_args())
    assert isinstance(model, Model)


def test_registered_ett_single_feature_schema():
    args = _build_args()
    args.features = "S"
    args.enc_in = 1
    args.dec_in = 1
    args.c_out = 1
    schema = resolve_tft_schema(args, feature_names=("OT",))
    assert schema.observed_positions == (0,)
    assert schema.target_positions == (0,)


def test_explicit_schema_overrides_registry():
    args = _build_args()
    args.tft_observed_pos = [1, 3]
    args.tft_target_pos = [3]
    args.c_out = 1
    schema = resolve_tft_schema(args)
    assert schema.observed_positions == (1, 3)
    assert schema.target_positions == (3,)


def test_static_observed_overlap_rejected():
    args = _build_args()
    args.tft_observed_pos = [0, 1]
    args.tft_static_pos = [1]
    with pytest.raises(ValueError, match="overlap"):
        resolve_tft_schema(args)


def test_target_must_be_observed():
    args = _build_args()
    args.tft_observed_pos = [0, 1]
    args.tft_target_pos = [2]
    args.c_out = 1
    with pytest.raises(ValueError, match="historically observed"):
        resolve_tft_schema(args)


def test_encoder_mark_length_must_equal_sequence_length():
    args = _build_args()
    model = Model(args)
    x_enc = __import__("torch").randn(2, args.seq_len, args.enc_in)
    x_mark_enc = __import__("torch").randn(2, args.seq_len - 1, 4)
    x_dec = __import__("torch").randn(2, args.label_len + args.pred_len, args.c_out)
    x_mark_dec = __import__("torch").randn(2, args.label_len + args.pred_len, 4)
    with pytest.raises(ValueError, match="x_mark_enc time length must equal seq_len"):
        model(x_enc, x_mark_enc, x_dec, x_mark_dec)


def test_detailed_frequency_known_length():
    args = _build_args()
    args.freq = "15min"
    schema = resolve_tft_schema(args)
    assert len(schema.known_feature_names) == 5


def test_custom_known_requires_names_and_length():
    args = _build_args()
    args.tft_allow_custom_known = True
    args.tft_known_len = 3
    args.tft_known_feature_names = None
    with pytest.raises(ValueError, match="tft_known_feature_names is required"):
        resolve_tft_schema(args)


def test_duplicate_quantiles_rejected():
    args = _build_args()
    args.tft_use_quantile_head = True
    args.tft_output_mode = "joint"
    args.tft_output_quantiles = [0.1, 0.5, 0.5]
    with pytest.raises(ValueError, match="duplicates"):
        Model(args)


def test_quantile_mode_requires_median():
    args = _build_args()
    args.tft_use_quantile_head = True
    args.tft_output_mode = "quantile"
    args.tft_output_quantiles = [0.1, 0.9]
    with pytest.raises(ValueError, match="requires quantile level 0.5"):
        Model(args)


def _build_static_args(use_revin=False, backbone="lstm"):
    args = _build_args()
    args.data = "custom_static_contracts"
    args.features = "M"
    args.seq_len = 6
    args.label_len = 3
    args.pred_len = 2
    args.enc_in = 3
    args.dec_in = 1
    args.c_out = 1
    args.d_model = 8
    args.dropout = 0.0
    args.tft_observed_pos = [0, 1]
    args.tft_static_pos = [2]
    args.tft_target_pos = [0]
    args.tft_use_revin = use_revin
    args.tft_revin_affine = use_revin
    args.tft_temporal_backbone = backbone
    return args


def _make_tft_io(args, batch_size=2):
    x_enc = torch.zeros(batch_size, args.seq_len, args.enc_in)
    x_enc[:, :, 0] = torch.linspace(0.0, 1.0, steps=args.seq_len)
    x_enc[:, :, 1] = torch.linspace(1.0, 2.0, steps=args.seq_len)
    x_dec = torch.zeros(batch_size, args.label_len + args.pred_len, args.c_out)
    x_mark_enc = torch.zeros(batch_size, args.seq_len, 4)
    x_mark_dec = torch.zeros(batch_size, args.label_len + args.pred_len, 4)
    return x_enc, x_mark_enc, x_dec, x_mark_dec


class _RecordingStaticEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.last_input = None

    def forward(self, x, x_mark):
        self.last_input = x.detach().clone()
        return x.repeat(1, 1, self.d_model)


class _RecordingHistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.recorded_state = None

    def forward(self, history_input, state):
        self.recorded_state = state
        return history_input, state


class _RecordingFutureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.recorded_state = None

    def forward(self, future_input, state):
        self.recorded_state = state
        return future_input, state


class _RecordingHybridBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.recorded_state = None

    def forward(self, temporal_input, state=None):
        self.recorded_state = state
        return temporal_input, state


def test_static_values_survive_manual_normalization():
    args = _build_static_args(use_revin=False)
    model = Model(args).eval()
    recorder = _RecordingStaticEmbedding(args.d_model)
    model.embedding.static_embedding[0] = recorder
    x_enc, x_mark_enc, x_dec, x_mark_dec = _make_tft_io(args)
    x_enc[0, :, 2] = 5.0
    x_enc[1, :, 2] = 50.0

    with torch.no_grad():
        model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    observed_static = recorder.last_input.squeeze(-1).squeeze(-1)
    assert torch.equal(observed_static, torch.tensor([5.0, 50.0]))


def test_static_values_survive_revin():
    args = _build_static_args(use_revin=True)
    model = Model(args).eval()
    recorder = _RecordingStaticEmbedding(args.d_model)
    model.embedding.static_embedding[0] = recorder
    x_enc, x_mark_enc, x_dec, x_mark_dec = _make_tft_io(args)
    x_enc[0, :, 2] = 5.0
    x_enc[1, :, 2] = 50.0

    with torch.no_grad():
        model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    observed_static = recorder.last_input.squeeze(-1).squeeze(-1)
    assert torch.equal(observed_static, torch.tensor([5.0, 50.0]))


def test_static_context_changes_with_entity_value():
    args = _build_static_args(use_revin=False)
    model = Model(args).eval()
    x_enc, x_mark_enc, x_dec, x_mark_dec = _make_tft_io(args)
    x_enc[0, :, 2] = 5.0
    x_enc[1, :, 2] = 50.0

    with torch.no_grad():
        payload = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation=True)

    static_context = payload["static_context"]["c_s"]
    assert not torch.allclose(static_context[0], static_context[1])


def test_nonconstant_static_feature_rejected():
    args = _build_static_args(use_revin=False)
    model = Model(args).eval()
    x_enc, x_mark_enc, x_dec, x_mark_dec = _make_tft_io(args, batch_size=1)
    x_enc[0, :, 2] = torch.tensor([5.0, 5.0, 5.0, 6.0, 5.0, 5.0])

    with pytest.raises(ValueError, match="must remain constant"):
        model(x_enc, x_mark_enc, x_dec, x_mark_dec)


def test_static_target_overlap_rejected():
    args = _build_static_args(use_revin=False)
    args.tft_observed_pos = [0]
    args.tft_static_pos = [0]
    args.tft_target_pos = [0]
    with pytest.raises(ValueError, match="overlap"):
        resolve_tft_schema(args)


def test_lstm_receives_hidden_then_cell_context():
    args = _build_static_args(use_revin=False, backbone="lstm")
    model = Model(args).eval()
    history_recorder = _RecordingHistoryEncoder()
    future_recorder = _RecordingFutureEncoder()
    model.temporal_fusion_decoder.layers[0].history_encoder = history_recorder
    model.temporal_fusion_decoder.layers[0].future_encoder = future_recorder
    x_enc, x_mark_enc, x_dec, x_mark_dec = _make_tft_io(args, batch_size=1)
    x_enc[0, :, 2] = 7.0

    with torch.no_grad():
        payload = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation=True)

    expected_hidden = payload["static_context"]["c_h"].unsqueeze(0)
    expected_cell = payload["static_context"]["c_c"].unsqueeze(0)
    recorded_hidden, recorded_cell = history_recorder.recorded_state
    assert torch.allclose(recorded_hidden, expected_hidden)
    assert torch.allclose(recorded_cell, expected_cell)
    future_hidden, future_cell = future_recorder.recorded_state
    assert torch.allclose(future_hidden, expected_hidden)
    assert torch.allclose(future_cell, expected_cell)


def test_hybrid_receives_hidden_then_cell_context():
    args = _build_static_args(use_revin=False, backbone="hybrid_tcn_lstm")
    model = Model(args).eval()
    hybrid_recorder = _RecordingHybridBackbone()
    model.temporal_fusion_decoder.layers[0].temporal_backbone = hybrid_recorder
    x_enc, x_mark_enc, x_dec, x_mark_dec = _make_tft_io(args, batch_size=1)
    x_enc[0, :, 2] = 9.0

    with torch.no_grad():
        payload = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation=True)

    expected_hidden = payload["static_context"]["c_h"].unsqueeze(0)
    expected_cell = payload["static_context"]["c_c"].unsqueeze(0)
    recorded_hidden, recorded_cell = hybrid_recorder.recorded_state
    assert torch.allclose(recorded_hidden, expected_hidden)
    assert torch.allclose(recorded_cell, expected_cell)


def test_static_interpretation_payload_preserved():
    args = _build_static_args(use_revin=False)
    args.tft_feature_names = ["target", "dynamic_aux", "static_entity"]
    model = Model(args).eval()
    x_enc, x_mark_enc, x_dec, x_mark_dec = _make_tft_io(args, batch_size=1)
    x_enc[0, :, 2] = 11.0

    with torch.no_grad():
        payload = model(x_enc, x_mark_enc, x_dec, x_mark_dec, return_interpretation=True)

    assert payload["static_vsn_weights"] is not None
    assert payload["static_graph_attention"] is not None
    assert payload["static_feature_names"] == ("static_entity",)
    for context_key in ("c_s", "c_c", "c_h", "c_e"):
        assert context_key in payload["static_vsn_weights"]
        assert context_key in payload["static_graph_attention"]
        context_weights = payload["static_vsn_weights"][context_key]
        assert context_weights is not None
        assert context_weights.shape[-1] == 1
