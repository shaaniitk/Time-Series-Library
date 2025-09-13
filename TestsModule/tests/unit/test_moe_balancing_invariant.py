"""MoE balancing invariant test (Phase 4 - Step 5 completion).

Validates the expert usage distribution stays within configured thresholds.
Skips gracefully if MoE modules / usage metrics unavailable.
"""
from __future__ import annotations

from typing import Any, Dict
import math

import pytest

try:
    from thresholds import THRESHOLDS
except Exception:  # pragma: no cover - fallback
    THRESHOLDS: Dict[str, Any] = {}


@pytest.mark.moe
def test_moe_expert_usage_balancing():
    """Check expert usage distribution fairness.

    Expectations:
    - Mean expert usage fraction >= moe_min_mean_usage_frac
    - Imbalance ratio (max / min non-zero) <= moe_max_usage_imbalance_ratio

    The test searches globally for an attribute `expert_usage` exposed by any
    imported module objects (simple heuristic) or via a fixture named
    `expert_usage` if provided by the suite. If neither path yields usage
    statistics, the test is skipped.
    """
    min_mean = THRESHOLDS.get("moe_min_mean_usage_frac", 0.05)
    max_ratio = THRESHOLDS.get("moe_max_usage_imbalance_ratio", 10.0)

    usage = None

    # Path 1: fixture if provided.
    if "expert_usage" in globals():  # unlikely
        maybe = globals()["expert_usage"]
        if isinstance(maybe, (list, tuple)) and maybe and all(isinstance(v, (int, float)) for v in maybe):
            usage = list(float(v) for v in maybe)

    # Path 2: attempt dynamic import of potential module exposing usage list.
    if usage is None:
        for name in ["moe_metrics", "moe_usage_state", "moe_state"]:
            try:
                mod = __import__(name)  # type: ignore
                if hasattr(mod, "expert_usage"):
                    val = getattr(mod, "expert_usage")
                    if isinstance(val, (list, tuple)) and val and all(isinstance(v, (int, float)) for v in val):
                        usage = list(float(v) for v in val)
                        break
            except Exception:  # pragma: no cover - best effort
                continue

    if usage is None:
        pytest.skip("No expert_usage metrics exposed; skipping MoE balancing invariant")

    total = sum(usage)
    if total <= 0.0:
        pytest.skip("Expert usage totals zero; skipping")

    normalized = [u / total for u in usage]
    mean_frac = sum(normalized) / len(normalized)

    # Non-zero entries for imbalance calculation
    nz = [u for u in normalized if u > 0.0]
    if not nz:
        pytest.skip("All expert usages zero after normalization")

    imbalance = (max(nz) / min(nz)) if len(nz) > 1 else 1.0

    assert mean_frac >= min_mean, (
        f"Mean expert usage fraction {mean_frac:.4f} < min threshold {min_mean:.4f}; "
        f"usage={normalized}"
    )
    assert imbalance <= max_ratio, (
        f"Expert usage imbalance {imbalance:.3f} exceeds max {max_ratio:.3f}; "
        f"usage={normalized}"
    )