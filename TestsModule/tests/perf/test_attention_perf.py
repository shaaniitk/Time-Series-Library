"""Performance sanity tests for selected attention mechanisms.

Marked with @pytest.mark.perf to allow optional invocation.
Keeps runtime modest; not a benchmark harness.
"""
from __future__ import annotations

import time
import pytest
import torch

# Use unified registry helper; populate registrations explicitly
from layers.modular.core import get_attention_component  # type: ignore
import layers.modular.core.register_components  # noqa: F401  # populate registry side-effects

ATTN_UNDER_TEST = [
    ("fourier_attention", {}),
    ("linear_attention", {}),
]

@pytest.mark.perf
@pytest.mark.parametrize("name,kwargs", ATTN_UNDER_TEST)
def test_attention_forward_time(name: str, kwargs: dict) -> None:
    # Skip gracefully if component isn't registered (e.g., linear_attention not present)
    try:
        attn = get_attention_component(name, d_model=128, n_heads=4, **kwargs)
    except Exception:
        pytest.skip(f"Attention component '{name}' not available")

    x = torch.randn(2, 64, 128)
    # Warmup
    _ = attn(x, x, x)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    start = time.time()
    for _ in range(3):
        out, _ = attn(x, x, x)
    elapsed = (time.time() - start) / 3

    assert out.shape == x.shape
    # Loose upper bound to catch pathological slowdowns; adjust as needed
    assert elapsed < 2.0, f"Forward too slow: {elapsed:.3f}s"
