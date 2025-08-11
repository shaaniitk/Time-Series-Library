"""Unified Factory Modular Tests

Replaces legacy root-level unified factory script with focused pytest cases.
"""
from __future__ import annotations

import pytest
import torch

try:
    from models.unified_autoformer_factory import (
        UnifiedModelInterface,
        create_autoformer,
        list_available_models,
        compare_implementations,
    )
except Exception:  # pragma: no cover - optional path
    UnifiedModelInterface = object  # type: ignore
    create_autoformer = None  # type: ignore
    list_available_models = lambda: {}  # type: ignore
    compare_implementations = lambda *a, **k: {}  # type: ignore


@pytest.fixture(scope="module")
def base_config() -> dict:
    # Align with legacy unified factory expectations (7 channels) to avoid shape mismatch
    return {
        'seq_len': 32,
        'pred_len': 8,
        'label_len': 16,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'd_model': 64,
        'n_heads': 4,
        'd_ff': 256,
        'e_layers': 1,
        'd_layers': 1,
        'dropout': 0.1,
        'activation': 'gelu'
    }


def test_model_listing_available() -> None:
    models = list_available_models()
    assert isinstance(models, dict)
    assert 'custom' in models


@pytest.mark.parametrize("model_key,framework", [('enhanced', 'custom')])
def test_model_creation(model_key: str, framework: str, base_config: dict) -> None:
    model = create_autoformer(model_key, base_config, framework=framework)
    assert isinstance(model, UnifiedModelInterface)
    info = model.get_model_info()
    assert 'framework_type' in info


def test_prediction_interface(base_config: dict) -> None:
    model = create_autoformer('enhanced', base_config, framework='custom')
    batch = 2
    x_enc = torch.randn(batch, base_config['seq_len'], base_config['enc_in'])
    x_mark_enc = torch.randn(batch, base_config['seq_len'], 4)
    x_dec = torch.randn(batch, base_config['label_len'] + base_config['pred_len'], base_config['dec_in'])
    x_mark_dec = torch.randn(batch, base_config['label_len'] + base_config['pred_len'], 4)
    with torch.no_grad():
        out = model.predict(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert out.shape[0] == batch
    assert out.shape[1] == base_config['pred_len']
    assert out.shape[2] == base_config['c_out']
    assert not torch.isnan(out).any()


def test_compare_implementations(base_config: dict) -> None:
    comparison = compare_implementations('enhanced', base_config)
    assert isinstance(comparison, dict)
    assert comparison
