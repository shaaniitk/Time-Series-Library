from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from run import (
    build_parser,
    execute_training_runs,
    normalize_args,
    resolve_run_args,
)
from utils.losses import QuantileLoss
from utils.tft_config import apply_tft_profile


def _dataset_args(**overrides):
    values = {
        "model": "TemporalFusionTransformer",
        "tft_extension_semantics_version": 2,
        "tft_position_source": "explicit_argument",
        "tft_position_unit": "calendar_days",
        "augmentation_ratio": 0,
        "extra_tag": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_irregular_market_series(path):
    dates = pd.bdate_range("2024-01-01", periods=40).delete([5, 13, 24])
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature": np.sin(np.arange(len(dates), dtype=np.float64)),
            "OT": np.cos(np.arange(len(dates), dtype=np.float64)),
        }
    )
    frame.to_csv(path, index=False)
    return frame


def test_custom_dataset_emits_calendar_coordinates_and_boolean_validity(tmp_path):
    frame = _write_irregular_market_series(tmp_path / "market.csv")
    dataset = Dataset_Custom(
        _dataset_args(),
        root_path=str(tmp_path),
        flag="train",
        size=[4, 2, 3],
        features="M",
        data_path="market.csv",
        target="OT",
        scale=False,
        timeenc=1,
        freq="d",
    )

    sample = dataset[0]
    assert len(sample) == 6
    positions, valid_mask = sample[-2:]
    timestamp_ns = pd.DatetimeIndex(frame["date"]).asi8
    expected = (timestamp_ns[:7] - timestamp_ns[0]) / (24 * 60 * 60 * 1e9)
    np.testing.assert_array_equal(positions, expected)
    assert valid_mask.dtype == np.bool_
    assert valid_mask.all()
    assert dataset.fold_manifest["temporal_coordinates"]["unit"] == "calendar_days"
    assert dataset.fold_manifest["temporal_coordinates"]["source"] == "explicit_argument"

    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert len(batch) == 6
    assert batch[-2].dtype == torch.float64
    assert batch[-2].shape == (2, 7)
    assert batch[-1].dtype == torch.bool


def test_legacy_custom_dataset_sample_shape_is_unchanged(tmp_path):
    _write_irregular_market_series(tmp_path / "market.csv")
    dataset = Dataset_Custom(
        _dataset_args(tft_extension_semantics_version=1),
        root_path=str(tmp_path),
        flag="train",
        size=[4, 2, 3],
        features="M",
        data_path="market.csv",
        target="OT",
        scale=False,
        timeenc=1,
        freq="d",
    )
    assert len(dataset[0]) == 4
    assert not hasattr(dataset, "data_temporal_positions")


def test_experiment_forwards_coordinate_tensors_without_wrapping_before_scatter():
    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.kwargs = None

        def forward(self, x, x_mark, dec, dec_mark, **kwargs):
            self.kwargs = kwargs
            return x

    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    experiment.args = SimpleNamespace(model="TemporalFusionTransformer")
    experiment._tft_target_positions = None
    experiment.model = Recorder()
    values = torch.randn(2, 4, 3)
    marks = torch.randn(2, 4, 2)
    decoder = torch.randn(2, 5, 3)
    decoder_marks = torch.randn(2, 5, 2)
    positions = torch.arange(7, dtype=torch.float64).expand(2, -1)
    valid = torch.ones(2, 7, dtype=torch.bool)

    output = experiment._forward_model(
        values,
        marks,
        decoder,
        decoder_marks,
        positions,
        valid,
    )

    assert output is values
    assert experiment.model.kwargs["temporal_positions"] is positions
    assert experiment.model.kwargs["temporal_valid_mask"] is valid


def test_masked_point_and_quantile_losses_ignore_invalid_nan_targets():
    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    experiment._tft_target_positions = (0,)
    experiment._tft_output_mode = "point"
    outputs = torch.tensor([[[2.0], [999.0], [5.0]]])
    target = torch.tensor([[[1.0], [float("nan")], [3.0]]])
    valid = torch.tensor([[True, False, True]])
    point_loss = experiment._compute_supervised_loss(
        outputs, target, nn.MSELoss(), None, valid_mask=valid
    )
    assert point_loss.item() == pytest.approx(2.5)

    quantile_forecast = outputs.unsqueeze(2).expand(-1, -1, 3, -1)
    quantile_loss = QuantileLoss([0.1, 0.5, 0.9])(
        quantile_forecast, target, valid_mask=valid
    )
    assert torch.isfinite(quantile_loss)


def test_row_index_policy_requires_user_or_builtin_dataset_declaration():
    base = {
        "model": "TemporalFusionTransformer",
        "task_name": "long_term_forecast",
        "data": "custom",
        "features": "MS",
        "enc_in": 2,
        "c_out": 1,
        "freq": "d",
        "tft_observed_pos": [0, 1],
        "tft_target_pos": [1],
        "tft_extension_semantics_version": 2,
    }
    with pytest.raises(ValueError, match="declared_regular_sampling=True"):
        apply_tft_profile(SimpleNamespace(**base))

    declared = apply_tft_profile(
        SimpleNamespace(**base, tft_declared_regular_sampling=True)
    )
    assert declared.tft_regular_sampling_declaration_source == "user"

    ett = apply_tft_profile(SimpleNamespace(**{**base, "data": "ETTh1"}))
    assert ett.tft_declared_regular_sampling is True
    assert (
        ett.tft_regular_sampling_declaration_source
        == "built_in_dataset_contract"
    )


def test_explicit_calendar_coordinates_run_through_train_validation_and_test(
    tmp_path,
    monkeypatch,
):
    _write_irregular_market_series(tmp_path / "market.csv")
    argv = (
        "--task_name long_term_forecast --is_training 1 "
        "--model_id sr02-calendar-e2e --model TemporalFusionTransformer "
        f"--data custom --root_path {tmp_path} --data_path market.csv "
        "--features MS --target OT --freq d "
        "--seq_len 8 --label_len 4 --pred_len 2 "
        "--enc_in 2 --dec_in 2 --c_out 1 "
        "--tft_observed_pos 0,1 --tft_target_pos 1 "
        "--tft_position_source explicit_argument "
        "--tft_position_unit calendar_days "
        "--d_model 8 --n_heads 2 --e_layers 1 --d_layers 1 "
        "--tft_profile extended_safe --tft_temporal_backbone lstm "
        "--dropout 0.0 --train_epochs 1 --batch_size 16 "
        "--num_workers 0 --patience 1 "
        f"--checkpoints {tmp_path / 'checkpoints'} --seed 2718 --no_use_gpu"
    ).split()
    args = normalize_args(build_parser().parse_args(argv))
    settings = execute_training_runs(args, Exp_Long_Term_Forecast)
    assert len(settings) == 1
    checkpoint = tmp_path / "checkpoints" / settings[0] / "checkpoint.pth"
    assert checkpoint.is_file()

    eval_args = SimpleNamespace(**vars(args))
    eval_args.is_training = 0
    eval_run_args, _ = resolve_run_args(eval_args, 0)
    evaluation = Exp_Long_Term_Forecast(eval_run_args)
    monkeypatch.chdir(tmp_path)
    evaluation.test(settings[0], test=1)
    assert (tmp_path / "results" / settings[0] / "metrics.npy").is_file()
