import copy
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import run
from astro.known import clear_cache, prepare_astro_known
from astro.rules.registry import RuleContractError
from astro.synthetic_study import write_synthetic_study
from data_provider.data_loader import Dataset_Custom, Dataset_PlanetaryMarket
from data_provider.folds import (
    AST_H01_DEVELOPMENT_FOLDS,
    FoldContractError,
    fold_borders,
    resolve_fold,
)

RULESET_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'astrology', 'ast_v1_core.json'
)
SEQ_LEN = 32


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    return write_synthetic_study(str(tmp_path_factory.mktemp("astro_study")))


def _args(study, **overrides):
    argv = [
        '--task_name', 'long_term_forecast', '--is_training', '1',
        '--model', 'TemporalFusionTransformer', '--data', 'planetary_market',
        '--root_path', study['root_path'], '--data_path', study['data_path'],
        '--target', 'log_Close', '--features', 'MS', '--enc_in', '4', '--dec_in', '4',
        '--c_out', '1', '--tft_observed_pos', '0,1,2,3', '--tft_target_pos', '3',
        '--freq', 'b', '--seq_len', str(SEQ_LEN), '--label_len', '8', '--pred_len', '1',
        '--d_model', '16', '--n_heads', '2', '--e_layers', '1', '--d_layers', '1',
        '--tft_profile', 'extended_safe', '--tft_extension_semantics_version', '2',
        '--tft_declared_regular_sampling', '--tft_position_unit', 'steps',
        '--tft_position_source', 'row_index', '--model_id', 'astro_test',
        '--astro_ruleset', RULESET_PATH,
        '--astro_ephemeris_path', study['ephemeris_path'],
        '--astro_ephemeris_manifest', study['manifest_path'],
    ]
    for key, value in overrides.items():
        argv += [f'--{key}', str(value)]
    clear_cache()
    return run.normalize_args(run.build_parser().parse_args(argv))


def _dataset(args, flag):
    return Dataset_PlanetaryMarket(
        args, args.root_path, flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features, data_path=args.data_path, target=args.target,
        timeenc=1, freq=args.freq,
    )


def test_layout_written_before_model_construction(study):
    args = _args(study)
    assert args.tft_allow_custom_known is True
    assert args.tft_known_len == len(args.tft_known_feature_names)
    assert args.tft_known_feature_names[:3] == ['DayOfWeek', 'DayOfMonth', 'DayOfYear']
    assert all(name.startswith('astro.') for name in args.tft_known_feature_names[3:])


def test_known_marks_width_matches_model_contract(study):
    args = _args(study)
    for flag in ('train', 'val', 'test'):
        dataset = _dataset(args, flag)
        _, _, seq_x_mark, seq_y_mark = dataset[0][:4]
        assert seq_x_mark.shape == (args.seq_len, args.tft_known_len)
        assert seq_y_mark.shape == (args.label_len + args.pred_len, args.tft_known_len)


def test_rule_channels_are_not_scaled(study):
    args = _args(study)
    dataset = _dataset(args, 'train')
    rules = dataset.data_stamp[:, 3:]
    names = args.tft_known_feature_names[3:]
    sin_col = names.index('astro.western.harmonic.jupiter_saturn_harmonics.h1_sin')
    cos_col = names.index('astro.western.harmonic.jupiter_saturn_harmonics.h1_cos')
    np.testing.assert_allclose(rules[:, sin_col] ** 2 + rules[:, cos_col] ** 2, 1.0, atol=1e-12)


def test_known_block_ignores_market_values(study, tmp_path):
    args = _args(study)
    baseline = _dataset(args, 'train').data_stamp.copy()

    market = pd.read_csv(os.path.join(study['root_path'], study['data_path']))
    for column in ('log_Open', 'log_High', 'log_Low', 'log_Close'):
        market[column] = np.random.default_rng(99).standard_normal(len(market)) * 50.0
    perturbed_root = str(tmp_path)
    market.to_csv(os.path.join(perturbed_root, study['data_path']), index=False)

    args_perturbed = _args({**study, 'root_path': perturbed_root})
    assert np.array_equal(_dataset(args_perturbed, 'train').data_stamp, baseline)


def test_split_dates_follow_fold_and_purge(study):
    args = _args(study, astro_fold='F2')
    train = _dataset(args, 'train')
    val = _dataset(args, 'val')
    fold = AST_H01_DEVELOPMENT_FOLDS['F2']
    last_train = pd.Timestamp(train.fold_manifest['last_forecast_timestamp'])
    first_eval = pd.Timestamp(val.fold_manifest['first_forecast_timestamp'])
    assert last_train <= pd.Timestamp(fold.train_end)
    assert first_eval >= pd.Timestamp(fold.eval_start)
    sessions = pd.bdate_range(last_train, first_eval)
    assert len(sessions) - 1 >= fold.purge_sessions


def test_expanding_training_windows_are_nested():
    dates = pd.Series(pd.bdate_range('1996-11-05', '2020-12-31'))
    ends = [fold_borders(dates, AST_H01_DEVELOPMENT_FOLDS[name], SEQ_LEN, 1)[1][0]
            for name in ('F1', 'F2', 'F3', 'F4')]
    assert ends == sorted(ends) and len(set(ends)) == 4


def test_purge_trims_training_when_gap_is_short():
    dates = pd.Series(pd.bdate_range('2000-01-03', '2012-12-31'))
    fold = AST_H01_DEVELOPMENT_FOLDS['F1']
    tight = type(fold)('tight', '2008-03-15', '2008-04-01', '2009-12-31')
    border1s, border2s = fold_borders(pd.Series(dates), tight, SEQ_LEN, 1)
    eval_start = border1s[1] + SEQ_LEN
    assert eval_start - border2s[0] >= tight.purge_sessions


def test_locked_holdout_requires_unlock():
    with pytest.raises(FoldContractError, match="locked holdout"):
        resolve_fold('HOLDOUT')
    assert resolve_fold('HOLDOUT', unlock_holdout=True).is_locked_holdout


def test_zero_and_null_arms_share_layout(study):
    real = _args(study)
    zero = _args(study, astro_arm='zero')
    null = _args(study, astro_arm='null:date_shift:days=1461')
    assert real.tft_known_feature_names == zero.tft_known_feature_names == null.tft_known_feature_names
    assert not _dataset(zero, 'train').data_stamp[:, 3:].any()


def test_unknown_arm_rejected(study):
    with pytest.raises(RuleContractError, match="Unknown astro_arm"):
        _args(study, astro_arm='null:nonsense')


def test_explicit_conflicting_known_names_rejected(study):
    with pytest.raises(RuleContractError, match="disagrees"):
        _args(study, tft_known_feature_names='a,b,c')


def test_custom_dataset_split_unchanged(tmp_path):
    frame = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=200, freq='h').astype(str),
        'x': np.arange(200.0),
        'OT': np.arange(200.0) * 2,
    })
    frame.to_csv(tmp_path / 'toy.csv', index=False)
    args = type('A', (), {'model': 'DLinear'})()
    dataset = Dataset_Custom(args, str(tmp_path), flag='val', size=[24, 12, 12],
                             features='M', data_path='toy.csv', target='OT', timeenc=1, freq='h')
    num_train, num_test = int(200 * 0.7), int(200 * 0.2)
    assert len(dataset.data_x) == (num_train + (200 - num_train - num_test)) - (num_train - 24)
    assert dataset.data_stamp.shape[1] == 4
