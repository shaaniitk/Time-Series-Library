"""Performance timing harness for invariant tests (Phase 4 - Step 1).

Collects rough wall-clock timings for a representative subset of invariant tests
so we can later drive automatic threshold tuning / slow-test detection.

Does not assert on timing yet; only prints JSON summary (consumed by summary plugin).
"""
from __future__ import annotations

import time
import json
import importlib
import pytest

SAMPLED_TESTS = [
    ("tests.invariants.test_attention_invariants", "test_attention_row_stochastic_and_mask"),
    ("tests.invariants.test_decomposition_invariants", "test_decomposition_reconstruction"),
    ("tests.invariants.test_output_head_invariants", "test_probabilistic_head_logvar_and_std_consistency"),
    ("tests.invariants.test_adapter_invariants", "test_adapter_covariate_effect_present"),
]


def _run_callable(mod_name: str, fn_name: str):
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    start = time.perf_counter()
    fn()  # direct call (tests are simple & self-contained)
    return time.perf_counter() - start


@pytest.mark.perf
def test_invariant_sample_timings():  # type: ignore
    timings = {}
    for mod, fn in SAMPLED_TESTS:
        try:
            dt = _run_callable(mod, fn)
            timings[f"{mod}::{fn}"] = dt
        except Exception as e:  # pragma: no cover - timing diagnostic path
            timings[f"{mod}::{fn}"] = f"ERROR: {e}"  # record failure for visibility
    print("[INVARIANT_TIMINGS] " + json.dumps(timings))

__all__ = []
