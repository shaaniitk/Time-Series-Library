import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from astro.report.arm_study import (
    ArmResult,
    block_bootstrap_ci,
    family_importance,
    paired_delta,
    verdict,
)
from astro.rules.registry import RuleContractError
from astro.rules.schema import RuleSet

RULESET_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'astrology', 'ast_v1_core.json'
)
DATES = pd.bdate_range('2010-01-01', periods=200)


def _arm(arm, fold, error, seed=2, noise=0.0):
    rng = np.random.default_rng(abs(hash((arm, fold, seed))) % 2**32)
    abs_err = np.full(len(DATES), error) + noise * rng.random(len(DATES))
    return ArmResult(arm, fold, seed, pd.DataFrame({
        'date': DATES, 'pred': 0.0, 'true': 0.0, 'abs_err': abs_err,
    }))


def _study(real_errors, zero_error=1.0, null_error=1.0):
    results = []
    for fold, real_error in real_errors.items():
        results += [
            _arm('real', fold, real_error),
            _arm('zero', fold, zero_error),
            _arm('null:date_shift:days=1461', fold, null_error),
        ]
    return results


def test_proceed_requires_three_of_four_folds():
    passing = verdict(_study({'F1': 0.9, 'F2': 0.9, 'F3': 0.9, 'F4': 1.1}))
    assert passing['decision'] == 'PROCEED'
    assert passing['winning_folds'] == ['F1', 'F2', 'F3']
    failing = verdict(_study({'F1': 0.9, 'F2': 0.9, 'F3': 1.1, 'F4': 1.1}))
    assert failing['decision'] == 'STOP'


def test_beating_zero_but_not_null_is_a_stop():
    results = _study({f: 0.9 for f in ('F1', 'F2', 'F3', 'F4')}, zero_error=1.0, null_error=0.85)
    assert verdict(results)['decision'] == 'STOP'


def test_missing_zero_arm_is_rejected():
    with pytest.raises(ValueError, match='real and zero'):
        verdict([_arm('real', 'F1', 0.9)])


def test_paired_delta_requires_identical_dates():
    shifted = _arm('zero', 'F1', 1.0)
    shifted.frame['date'] = shifted.frame['date'] + pd.Timedelta(days=1)
    with pytest.raises(RuntimeError, match='identical dates'):
        paired_delta(shifted, _arm('real', 'F1', 0.9))


def test_block_bootstrap_ci_brackets_the_mean():
    deltas = np.random.default_rng(0).normal(0.1, 1.0, 1000)
    low, high = block_bootstrap_ci(deltas)
    assert low < deltas.mean() < high
    assert low > -0.2 and high < 0.4


def test_family_importance_sign_and_ci():
    results = [
        _arm('real', 'F1', 0.9, noise=0.01),
        _arm('zero:retrograde', 'F1', 1.1, noise=0.01),
        _arm('zero:rashi', 'F1', 0.9, noise=0.01),
    ]
    rows = {row['family']: row for row in family_importance(results)}
    assert rows['retrograde']['mae_increase'] == pytest.approx(0.2, abs=0.01)
    low, high = rows['retrograde']['ci']
    assert low <= rows['retrograde']['mae_increase'] <= high
    assert abs(rows['rashi']['mae_increase']) < 0.01


def test_interaction_pairs_are_reserved():
    with open(RULESET_PATH, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    payload = copy.deepcopy(payload)
    payload['interactions'] = {'pairs': [['a', 'b']]}
    with pytest.raises(RuleContractError, match='reserved'):
        RuleSet.from_json(payload)
