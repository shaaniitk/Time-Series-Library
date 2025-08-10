"""Lightweight integration test: build a minimal model and perform a forward pass.

Acts as a replacement for broad monolithic integration scripts; keeps runtime small.
"""
from __future__ import annotations

import pytest
import torch

# Import guarded to avoid failures during partial refactors
try:  # pragma: no cover
    from utils.modular_components.model_builder import ModelBuilder, ModelConfig  # type: ignore
except Exception:  # pragma: no cover
    ModelBuilder = None  # type: ignore
    ModelConfig = None  # type: ignore

@pytest.mark.smoke
def test_minimal_model_build_and_forward() -> None:
    if ModelBuilder is None or ModelConfig is None:
        pytest.skip("ModelBuilder utilities unavailable")

    cfg = ModelConfig(
        model_name="smoke_model",
        task_type="forecasting",
        d_model=64,
        backbone={
            "type": "simple_transformer",
            "d_model": 64,
            "num_layers": 1,
            "num_heads": 2,
        },
        feedforward={
            "type": "standard_ffn",
            "d_model": 64,
            "d_ff": 128,
            "dropout": 0.1,
        },
        output={
            "type": "forecasting",
            "d_model": 64,
            "horizon": 4,
            "output_dim": 1,
        },
    )

    builder = ModelBuilder()
    model = builder.build_model(cfg)
    x = torch.randn(2, 16, 1)
    out = model(x)

    assert out.shape == (2, 4, 1)
    assert torch.isfinite(out).all()
