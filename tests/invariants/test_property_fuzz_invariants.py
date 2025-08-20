"""Property-based fuzz tests for core invariants (Phase 4 - Step 3).

Uses Hypothesis to randomly generate input sizes and ensures key invariants hold
(determinism for encoder fallback; adapter pass-through with zero covariates).
"""
from __future__ import annotations

import pytest
import torch
try:  # optional dependency
    from hypothesis import given, settings, strategies as st  # type: ignore
    _HYPOTHESIS_AVAILABLE = True
except Exception:  # pragma: no cover - dependency missing
    _HYPOTHESIS_AVAILABLE = False

from .thresholds import get_threshold


pytestmark = pytest.mark.skipif(not _HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")

@pytest.mark.usefixtures()
@settings(max_examples=20, deadline=None) if _HYPOTHESIS_AVAILABLE else (lambda f: f)  # type: ignore
@given(batch=st.integers(min_value=1, max_value=4), length=st.integers(min_value=8, max_value=96)) if _HYPOTHESIS_AVAILABLE else (lambda f: f)  # type: ignore
def test_structured_encoder_determinism_fuzz(batch: int, length: int):  # type: ignore
    try:
        from tests.invariants.test_encoder_invariants import _get_structured_encoder  # type: ignore
    except Exception:
        pytest.skip("Structured encoder helper unavailable")
    enc = _get_structured_encoder()
    enc.eval()
    x = torch.randn(batch, length, enc.get_d_model())
    with torch.no_grad():
        out1 = enc(x)
        out2 = enc(x)
    rel = torch.linalg.norm(out1 - out2) / (torch.linalg.norm(out1) + 1e-12)
    assert rel <= get_threshold("encoder_determinism_tol"), rel


@settings(max_examples=20, deadline=None) if _HYPOTHESIS_AVAILABLE else (lambda f: f)  # type: ignore
@given(batch=st.integers(min_value=1, max_value=3), length=st.integers(min_value=16, max_value=64)) if _HYPOTHESIS_AVAILABLE else (lambda f: f)  # type: ignore
def test_covariate_adapter_pass_through_fuzz(batch: int, length: int):  # type: ignore
    try:
        from tests.invariants.test_adapter_invariants import _get_covariate_adapter  # type: ignore
    except Exception:
        pytest.skip("Adapter helper unavailable")
    adapter = _get_covariate_adapter()
    adapter.eval()
    x = torch.randn(batch, length, 1)
    with torch.no_grad():
        fused = adapter._fuse_covariates(x, None)
    err = torch.linalg.norm(fused - x) / (torch.linalg.norm(x) + 1e-12)
    assert err <= get_threshold("adapter_pass_through_rel_err"), err

__all__ = []
