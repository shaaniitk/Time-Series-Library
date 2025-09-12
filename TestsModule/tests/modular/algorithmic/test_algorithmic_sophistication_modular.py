"""Algorithmic sophistication preservation tests.

Migrates core semantic assertions from legacy `test_phase2_validation.py` and
`test_algorithmic_sophistication.py` (k_predictor complexity, meta-learning fast weights,
Fourier advanced filtering) into concise pytest tests.
"""
from __future__ import annotations

import pytest
import torch

from utils_algorithm_adapters import (
    register_restored_algorithms,
    RestoredFourierConfig,
    RestoredAutoCorrelationConfig,
    RestoredMetaLearningConfig,
)
from layers.modular.core.registry import component_registry as _global_registry, ComponentFamily


@pytest.fixture(scope="module", autouse=True)
def _register_restored() -> None:  # pragma: no cover - one-off side effect
    register_restored_algorithms()


def _create(component: str, cfg):
    return _global_registry.create(component, ComponentFamily.ATTENTION, config=cfg.to_dict())


def test_fourier_restored_capabilities() -> None:
    cfg = RestoredFourierConfig(d_model=64, seq_len=48, num_heads=4, dropout=0.0)
    attn = _create('restored_fourier_attention', cfg)
    x = torch.randn(2, 48, 64)
    with torch.no_grad():
        out, weights = attn.apply_attention(x, x, x)
    assert out.shape == x.shape
    caps = attn.get_capabilities()
    assert 'sophisticated_frequency_filtering' in caps


def test_autocorrelation_k_predictor_complexity() -> None:
    cfg = RestoredAutoCorrelationConfig(d_model=64, num_heads=4, dropout=0.0)
    attn = _create('restored_autocorrelation_attention', cfg)
    x = torch.randn(2, 32, 64)
    with torch.no_grad():
        out, _ = attn.apply_attention(x, x, x)
    assert out.shape == x.shape
    k_pred = attn.autocorr_attention.k_predictor
    param_count = sum(p.numel() for p in k_pred.parameters())
    assert param_count > 100, f"k_predictor too small ({param_count} params)"


def test_meta_learning_fast_weights_presence() -> None:
    cfg = RestoredMetaLearningConfig(d_model=64, num_heads=4, dropout=0.0)
    attn = _create('restored_meta_learning_attention', cfg)
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        out, _ = attn.apply_attention(x, x, x)
    assert out.shape == x.shape
    assert hasattr(attn.meta_attention, 'fast_weights')
