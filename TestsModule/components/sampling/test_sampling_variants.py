from __future__ import annotations

"""Sampling component tests.

Validates registry listing and minimal forward API for deterministic/bayesian/dropout
samplers using tiny tensors. Lightweight and marked extended.
"""

import pytest
import torch

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from layers.modular.sampling.registry import SamplingRegistry, get_sampling_component  # type: ignore
except Exception:  # pragma: no cover
    SamplingRegistry = None  # type: ignore
    get_sampling_component = None  # type: ignore


def _has(name: str) -> bool:
    if SamplingRegistry is None:
        return False
    try:
        return name in SamplingRegistry.list_components()
    except Exception:
        return False


def test_sampling_registry_non_empty() -> None:
    if SamplingRegistry is None:
        pytest.skip("SamplingRegistry unavailable")
    names = SamplingRegistry.list_components()
    assert isinstance(names, list) and names


@pytest.mark.parametrize("name", ["deterministic", "bayesian", "dropout"])
def test_sampling_forward_api(name: str) -> None:
    if SamplingRegistry is None or get_sampling_component is None:
        pytest.skip("SamplingRegistry unavailable")
    if not _has(name):
        pytest.skip(f"{name} sampler not registered")

    sampler = get_sampling_component(name)

    # Build a minimal model forward callable matching expected signature
    class DummyModel:
        def __init__(self):
            self.training = False
        def train(self, mode: bool = True):
            self.training = mode
            return self
        def eval(self):
            self.training = False
            return self
        def __call__(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
            # echo decoder input shape as prediction
            return x_dec
        # Optional hooks used by BayesianSampling
        def set_quantile_levels(self, *_args, **_kwargs):
            pass
        def get_quantile_levels(self):
            return None

    model = DummyModel()
    model_forward = model.__call__

    # Minimal encoder/decoder tensors and time marks
    x_enc = torch.randn(2, 8, 3)
    x_mark_enc = torch.zeros(2, 8, 1)
    x_dec = torch.randn(2, 4, 3)
    x_mark_dec = torch.zeros(2, 4, 1)

    out = sampler(model_forward, x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert isinstance(out, dict) and 'prediction' in out
    pred = out['prediction']
    assert isinstance(pred, torch.Tensor)
    assert pred.shape == x_dec.shape
    assert torch.isfinite(pred).all()
