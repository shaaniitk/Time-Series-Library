import copy
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import run
from astro.known import clear_cache
from astro.synthetic_study import write_synthetic_study
from astro.torch.importance import AstroRuleGates, parse_rule_layout
from astro.torch.losses import prior_penalty, response_regularity
from models.TemporalFusionTransformer import Model
from utils.tft_config import compute_tft_config_digest

RULESET_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'astrology', 'ast_v1_core.json'
)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    return write_synthetic_study(str(tmp_path_factory.mktemp("astro_gates")), market_end="2012-12-31")


def _args(study, *extra):
    argv = [
        '--task_name', 'long_term_forecast', '--is_training', '1',
        '--model', 'TemporalFusionTransformer', '--data', 'planetary_market',
        '--root_path', study['root_path'], '--data_path', study['data_path'],
        '--target', 'log_Close', '--features', 'MS', '--enc_in', '4', '--dec_in', '4',
        '--c_out', '1', '--tft_observed_pos', '0,1,2,3', '--tft_target_pos', '3',
        '--freq', 'b', '--seq_len', '16', '--label_len', '4', '--pred_len', '1',
        '--d_model', '8', '--n_heads', '2', '--e_layers', '1', '--d_layers', '1', '--dropout', '0.0',
        '--tft_profile', 'extended_safe', '--tft_extension_semantics_version', '2',
        '--tft_declared_regular_sampling', '--tft_position_unit', 'steps',
        '--tft_position_source', 'row_index', '--model_id', 'astro_gate_test',
        '--astro_ruleset', RULESET_PATH,
        '--astro_ephemeris_path', study['ephemeris_path'],
        '--astro_ephemeris_manifest', study['manifest_path'],
        *extra,
    ]
    clear_cache()
    return run.normalize_args(run.build_parser().parse_args(argv))


def _batch(args, seed=0):
    generator = torch.Generator().manual_seed(seed)
    batch = 3
    x_enc = torch.randn(batch, args.seq_len, args.enc_in, generator=generator)
    x_dec = torch.randn(batch, args.label_len + args.pred_len, args.enc_in, generator=generator)
    mark_enc = torch.rand(batch, args.seq_len, args.tft_known_len, generator=generator)
    mark_dec = torch.rand(batch, args.label_len + args.pred_len, args.tft_known_len, generator=generator)
    return x_enc, mark_enc, x_dec, mark_dec


def _model(args, seed=11):
    torch.manual_seed(seed)
    return Model(args).eval()


def test_layout_parsing_groups_channels_by_rule():
    names = ['DayOfWeek', 'astro.vedic.drishti.r1.act_a60', 'astro.vedic.drishti.r1.act_a60.hl21',
             'astro.western.aspect.r2.act_a0']
    columns, rule_ids, index, families = parse_rule_layout(names)
    assert columns == [1, 2, 3]
    assert rule_ids == ['r1', 'r2']
    assert index == [0, 0, 1]
    assert families == ['drishti', 'aspect']


def test_gates_start_neutral_so_real_inputs_match_zero_arm(study):
    args = _args(study, '--tft_astro_rule_gates')
    model = _model(args)
    x_enc, mark_enc, x_dec, mark_dec = _batch(args)
    columns = model.astro_rule_gates.columns
    zero_enc, zero_dec = mark_enc.clone(), mark_dec.clone()
    zero_enc[:, :, columns] = 0.0
    zero_dec[:, :, columns] = 0.0
    with torch.no_grad():
        real = model(x_enc, mark_enc, x_dec, mark_dec)
        zeroed = model(x_enc, zero_enc, x_dec, zero_dec)
    assert torch.equal(real, zeroed)


def test_gates_receive_gradient_at_neutral_init(study):
    args = _args(study, '--tft_astro_rule_gates')
    model = _model(args).train()
    x_enc, mark_enc, x_dec, mark_dec = _batch(args)
    model(x_enc, mark_enc, x_dec, mark_dec).pow(2).mean().backward()
    assert model.astro_rule_gates.gates.grad is not None
    assert model.astro_rule_gates.gates.grad.abs().sum() > 0


def test_gates_add_one_parameter_per_rule_and_nothing_else(study):
    base = _args(study)
    gated = _args(study, '--tft_astro_rule_gates')
    plain_keys = set(_model(base).state_dict())
    gated_state = _model(gated).state_dict()
    extra = set(gated_state) - plain_keys
    assert extra == {'astro_rule_gates.gates'}
    assert gated_state['astro_rule_gates.gates'].numel() == 14


def test_gates_off_leave_digest_unchanged_and_on_change_it(study):
    base = _args(study)
    without_attribute = copy.copy(base)
    del without_attribute.tft_astro_rule_gates
    assert compute_tft_config_digest(base) == compute_tft_config_digest(without_attribute)
    assert _args(study, '--tft_astro_rule_gates').tft_config_digest != base.tft_config_digest


def test_family_importance_aggregates_absolute_gates():
    gates = AstroRuleGates(['astro.vedic.drishti.a.x', 'astro.vedic.drishti.b.x', 'astro.western.aspect.c.x'])
    with torch.no_grad():
        gates.gates.copy_(torch.tensor([0.4, -0.2, 0.5]))
    assert gates.family_importance() == pytest.approx({'drishti': 0.3, 'aspect': 0.5})


def test_prior_penalty_matches_declared_formula():
    gates = torch.tensor([0.5, -0.1], requires_grad=True)
    importance = torch.tensor([0.8, 0.2])
    l2 = torch.tensor([2.0, 0.0])
    l1 = torch.tensor([0.0, 0.5])
    penalty = prior_penalty(gates, importance, l2, l1)
    assert penalty.item() == pytest.approx(2.0 * 0.09 + 0.5 * 0.1)


def test_response_regularity_is_zero_when_output_ignores_astro():
    generator = torch.Generator().manual_seed(0)
    marks = torch.rand(2, 5, 6)
    columns = torch.tensor([3, 4, 5])
    calendar_only = lambda enc, dec: enc[:, :, :3].sum(dim=(1, 2)) + dec[:, :, :3].sum(dim=(1, 2))
    penalty = response_regularity(calendar_only, marks, marks.clone(), columns, 0.05, generator)
    assert penalty.item() == pytest.approx(0.0, abs=1e-10)


def test_response_regularity_scales_with_astro_sensitivity():
    columns = torch.tensor([3, 4, 5])
    marks = torch.rand(2, 5, 6)

    def penalty_for(scale):
        generator = torch.Generator().manual_seed(0)
        predict = lambda enc, dec: scale * enc[:, :, 3:].sum(dim=(1, 2))
        return response_regularity(predict, marks, marks.clone(), columns, 0.05, generator).item()

    assert penalty_for(2.0) == pytest.approx(4.0 * penalty_for(1.0), rel=1e-5)
    assert penalty_for(1.0) > 0


def test_zero_coefficients_bypass_astro_loss(study):
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    exp = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    exp.args = SimpleNamespace(astro_prior_coeff=0.0, astro_regularity_coeff=0.0)
    assert exp._astro_auxiliary_loss(None, None, None, None, None, None) is None


def test_prior_coefficient_without_gates_is_rejected(study):
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    exp = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    exp.args = SimpleNamespace(astro_prior_coeff=0.1, astro_regularity_coeff=0.0)
    exp.model = SimpleNamespace(astro_rule_gates=None)
    exp._astro_term_log = {}
    with pytest.raises(ValueError, match="tft_astro_rule_gates"):
        exp._astro_auxiliary_loss(None, None, None, None, None, None)
