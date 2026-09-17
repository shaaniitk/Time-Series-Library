import copy
import json
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from astro.compile.compiler import compile_ruleset
from astro.compile.nulls import assert_arms_matched, compile_null_arm
from astro.ephemeris.provider import EphemerisProvider, build_synthetic_ephemeris
from astro.ephemeris.validator import AYANAMSHA_COLUMN
from astro.rules.registry import OPERATOR_REGISTRY, RuleContractError
from astro.rules.schema import RuleSet

RULESET_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'astrology', 'ast_v1_core.json'
)

# Parameters exercising every registered operator.
OPERATOR_CASES = {
    "wrapped_separation": ({"from_body": "Sun", "to_body": "Moon"}, {}),
    "harmonic_phase": ({"from_body": "Jupiter", "to_body": "Saturn"}, {"harmonics": [1, 3]}),
    "aspect_activation": ({"from_body": "Mars", "to_body": "Venus"}, {"aspect_deg": [0, 90], "sigma_deg": 5.0}),
    "degree_orb_drishti": ({"from_body": "Saturn", "to_body": "Jupiter"}, {"aspect_deg": [60, 180], "sigma_deg": 6.0}),
    "whole_sign_drishti": ({"from_body": "Saturn", "to_body": "Sun"}, {"house_offsets": [3, 7]}),
    "rashi_cyclic": ({"body": "Jupiter"}, {}),
    "nakshatra_pada": ({"body": "Moon"}, {}),
    "ingress_pulse": ({"body": "Sun"}, {"width_days": 3.0}),
    "retrograde_state": ({"body": "Mercury"}, {}),
    "station_proximity": ({"body": "Mercury"}, {"speed_scale": 0.5}),
    "combustion": ({"body": "Mercury"}, {"width_deg": 8.0}),
    "node_axis": ({"body": "Moon"}, {"width_deg": 5.0}),
    "declination_parallel": ({"body_a": "Mars", "body_b": "Venus"}, {"orb_deg": 1.0}),
    "midpoint_activation": ({"body_a": "Sun", "body_b": "Moon", "target_body": "Saturn"}, {"orb_deg": 2.0}),
}


@pytest.fixture(scope="module")
def ephemeris():
    return build_synthetic_ephemeris(periods=3000)


@pytest.fixture(scope="module")
def provider(ephemeris):
    frame, manifest = ephemeris
    return EphemerisProvider(frame.copy(), manifest)


@pytest.fixture(scope="module")
def ruleset_payload():
    with open(RULESET_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def ruleset(ruleset_payload):
    return RuleSet.from_json(ruleset_payload)


def test_every_registered_operator_has_a_case():
    assert set(OPERATOR_CASES) == set(OPERATOR_REGISTRY)


@pytest.mark.parametrize("name", sorted(OPERATOR_CASES))
def test_operator_is_deterministic_bounded_and_complete(provider, name):
    spec = OPERATOR_REGISTRY[name]
    inputs, params = OPERATOR_CASES[name]
    spec.validate_call(inputs, params)
    emits = spec.resolve_emits(params)
    first = spec.fn(provider, inputs, params)
    second = spec.fn(provider, inputs, params)
    assert set(first) == {emit.key for emit in emits}
    for emit in emits:
        values = first[emit.key]
        assert values.shape == (len(provider.index),)
        assert np.isfinite(values).all()
        assert values.min() >= emit.low - 1e-12 and values.max() <= emit.high + 1e-12
        assert np.array_equal(values, second[emit.key])


@pytest.mark.parametrize("name", sorted(OPERATOR_CASES))
def test_operator_honors_declared_ayanamsha_invariance(ephemeris, name):
    frame, manifest = ephemeris
    rotated = frame.copy()
    rotated[AYANAMSHA_COLUMN] = np.mod(rotated[AYANAMSHA_COLUMN] + 7.0, 360.0)
    base = EphemerisProvider(frame.copy(), manifest)
    shifted = EphemerisProvider(rotated, manifest)
    spec = OPERATOR_REGISTRY[name]
    inputs, params = OPERATOR_CASES[name]
    # Rotation only moves sidereal longitudes, so probe the sidereal frame.
    if "frame" in spec.optional_inputs:
        inputs = {**inputs, "frame": "sidereal"}
    a = spec.fn(base, inputs, params)
    b = spec.fn(shifted, inputs, params)
    worst = max(float(np.abs(a[key] - b[key]).max()) for key in a)
    if spec.invariance == "invariant":
        assert worst < 1e-10
    else:
        # Guards against a vacuous test: covariant operators must move.
        assert worst > 1e-6


def test_ruleset_hash_ignores_key_order_but_tracks_values(ruleset_payload, ruleset):
    reordered = json.loads(json.dumps(ruleset_payload, sort_keys=True))
    assert RuleSet.from_json(reordered).ruleset_hash == ruleset.ruleset_hash
    changed = copy.deepcopy(ruleset_payload)
    changed["rules"][0]["params"]["sigma_deg"] = 7.0
    assert RuleSet.from_json(changed).ruleset_hash != ruleset.ruleset_hash


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda r: r.update(operator="eval"), "Unknown operator"),
        (lambda r: r["inputs"].update(to_body="Jupiter", code="__import__('os')"), "unknown inputs"),
        (lambda r: r.update(emits=["not_an_emit"]), "does not produce"),
        (lambda r: r.update(school="babylonian"), "school must be"),
        (lambda r: r.update(rule_id="Bad Id"), "rule_id must match"),
        (lambda r: r.update(response_bank={"half_life_days": [4]}), "preregistered"),
        (lambda r: r["prior"].update(importance=1.5), "importance"),
        (lambda r: r.update(unexpected=True), "unknown keys"),
    ],
)
def test_invalid_rules_rejected(ruleset_payload, mutation, message):
    payload = copy.deepcopy(ruleset_payload)
    mutation(payload["rules"][0])
    with pytest.raises((RuleContractError, ValueError), match=message):
        RuleSet.from_json(payload)


def test_duplicate_rule_ids_rejected(ruleset_payload):
    payload = copy.deepcopy(ruleset_payload)
    payload["rules"].append(copy.deepcopy(payload["rules"][0]))
    with pytest.raises(RuleContractError, match="Duplicate rule_id"):
        RuleSet.from_json(payload)


def test_channel_names_are_named_unique_and_stable(ruleset_payload, ruleset, provider):
    compiled = compile_ruleset(ruleset, provider)
    assert len(set(compiled.names)) == len(compiled.names)
    assert all(name.startswith("astro.") for name in compiled.names)
    assert not any(re.fullmatch(r"f\d+", name) for name in compiled.names)

    extended = copy.deepcopy(ruleset_payload)
    extended["rules"].append(
        {
            "rule_id": "venus_retrograde",
            "school": "both",
            "family": "retrograde",
            "operator": "retrograde_state",
            "inputs": {"body": "Venus"},
            "params": {},
        }
    )
    grown = compile_ruleset(RuleSet.from_json(extended), provider)
    assert grown.names[: compiled.channel_count] == compiled.names
    assert grown.channel_count > compiled.channel_count


def test_disabled_rule_drops_only_its_channels(ruleset_payload, ruleset, provider):
    payload = copy.deepcopy(ruleset_payload)
    payload["rules"][0]["enabled"] = False
    compiled = compile_ruleset(RuleSet.from_json(payload), provider)
    dropped = payload["rules"][0]["rule_id"]
    assert not any(f".{dropped}." in name for name in compiled.names)
    full = compile_ruleset(ruleset, provider)
    assert set(compiled.names) < set(full.names)


def test_response_bank_stays_in_range_and_smooths(ruleset, provider):
    compiled = compile_ruleset(ruleset, provider)
    base = compiled.names.index("astro.both.retrograde.mercury_retrograde.is_retrograde")
    smoothed = compiled.names.index("astro.both.retrograde.mercury_retrograde.is_retrograde.hl21")
    raw, slow = compiled.matrix[:, base], compiled.matrix[:, smoothed]
    assert slow.min() >= 0.0 and slow.max() <= 1.0
    assert np.abs(np.diff(slow)).max() < np.abs(np.diff(raw)).max()


def test_ruleset_requirements_checked_against_manifest(ruleset_payload, ephemeris):
    frame, manifest = ephemeris
    payload = copy.deepcopy(ruleset_payload)
    payload["requires_ephemeris"]["node_policy"] = "true"
    with pytest.raises(RuleContractError, match="node_policy"):
        compile_ruleset(RuleSet.from_json(payload), EphemerisProvider(frame.copy(), manifest))


@pytest.mark.parametrize("arm_index", [0, 1, 2, 3])
def test_null_arms_are_capacity_matched_and_decorrelated(ruleset, provider, arm_index):
    real = compile_ruleset(ruleset, provider)
    null = ruleset.null_arms[arm_index]
    arm = compile_null_arm(ruleset, provider, null, real=real)
    assert_arms_matched(real, arm)
    assert all(meta.arm == null.arm_name for meta in arm.channel_meta)
    assert not np.array_equal(arm.matrix, real.matrix)


def test_phase_randomize_preserves_marginals(ruleset, provider):
    real = compile_ruleset(ruleset, provider)
    null = next(n for n in ruleset.null_arms if n.type == "phase_randomize")
    arm = compile_null_arm(ruleset, provider, null, real=real)
    np.testing.assert_array_equal(np.sort(arm.matrix, axis=0), np.sort(real.matrix, axis=0))


def test_null_arm_seed_is_reproducible(ruleset, provider):
    real = compile_ruleset(ruleset, provider)
    null = next(n for n in ruleset.null_arms if n.type == "body_permute")
    first = compile_null_arm(ruleset, provider, null, real=real)
    second = compile_null_arm(ruleset, provider, null, real=real)
    assert np.array_equal(first.matrix, second.matrix)
