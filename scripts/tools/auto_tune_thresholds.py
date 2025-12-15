"""Auto threshold tuner (Phase 4 - Step 2).

Runs selected invariant metric computations multiple times to derive tighter
threshold suggestions (median + k * IQR heuristic) and prints a patch-like
suggestion (does not modify files automatically).
"""
from __future__ import annotations

import statistics
import json
from typing import Callable, Dict, List
import importlib

# Mapping: name -> (module, callable_name, threshold_key)
TARGET_METRICS = {
    "adapter_covariate_effect": (
        "tests.invariants.test_adapter_invariants", "test_adapter_covariate_effect_present", "adapter_covariate_effect_min_delta"
    ),
    "prob_68pct_coverage": (
        "tests.invariants.test_probabilistic_and_multiscale_invariants", "test_probabilistic_head_empirical_coverage", "output_prob_68pct_coverage_abs_err_max"
    ),
}

REPEATS = 5
IQR_MULTIPLIER = 1.5


def _capture_metric(run_fn: Callable) -> float:
    # Here we assume the test stores metric in a global or we adapt tests to expose.
    # Placeholder: just run function and return 0.0 (extend as needed with hooks).
    run_fn()
    return 0.0


def main():
    results: Dict[str, List[float]] = {k: [] for k in TARGET_METRICS}
    for name, (mod_name, fn_name, key) in TARGET_METRICS.items():
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        vals: List[float] = []
        for _ in range(REPEATS):
            val = _capture_metric(fn)
            vals.append(val)
        results[name] = vals
    suggestions = {}
    for name, arr in results.items():
        if not arr:
            continue
        med = statistics.median(arr)
        # IQR fallback to median if insufficient data
        iqr = statistics.quantiles(arr, n=4)[2] - statistics.quantiles(arr, n=4)[0] if len(arr) >= 4 else 0.0
        suggestions[name] = {"median": med, "iqr": iqr, "suggested": med + IQR_MULTIPLIER * iqr}
    print(json.dumps({"runs": results, "suggestions": suggestions}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
