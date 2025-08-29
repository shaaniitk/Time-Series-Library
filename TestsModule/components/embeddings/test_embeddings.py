from __future__ import annotations

"""Embedding component tests.

Covers registry presence via unified facade and minimal forward shape/grad for
Temporal, Value, Covariate, and Hybrid embeddings with tiny tensors. Lightweight
and marked extended.
"""

from typing import Dict, Optional, Any, Callable, Tuple

import pytest
import torch

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from utils.modular_components.registry import get_global_registry  # type: ignore
    from configs.schemas import EmbeddingConfig  # type: ignore
    from layers.modular.embedding.temporal_embedding import TemporalEmbedding
    from layers.modular.embedding.value_embedding import ValueEmbedding
    from layers.modular.embedding.covariate_embedding import CovariateEmbedding
    from layers.modular.embedding.hybrid_embedding import HybridEmbedding
except Exception:  # pragma: no cover
    get_global_registry = None  # type: ignore
    EmbeddingConfig = None  # type: ignore
    register_utils_embeddings = None  # type: ignore


def test_embedding_registry_non_empty() -> None:
    """Registry lists at least one embedding and has unique names."""
    if get_global_registry is None or register_utils_embeddings is None:
        pytest.skip("Registry unavailable")
    # Ensure embeddings are registered into the global registry
    register_utils_embeddings()
    reg = get_global_registry()
    comps = reg.list_components()
    names = comps.get("embedding", [])
    assert isinstance(names, list) and names
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "name,setup",
    [
        (
            "temporal_embedding",
            lambda cls: (
                cls(EmbeddingConfig(d_model=16, dropout=0.0)),
                {
                    "input_embeddings": torch.randn(2, 6, 16, requires_grad=True),
                    "temporal_features": {
                        "hour": torch.randint(0, 24, (2, 6)),
                        "day": torch.randint(1, 31, (2, 6)),
                        "weekday": torch.randint(0, 7, (2, 6)),
                        "month": torch.randint(1, 13, (2, 6)),
                    },
                    "positions": None,
                },
            ),
        ),
        (
            "value_embedding",
            lambda cls: (
                cls(EmbeddingConfig(d_model=16, dropout=0.0)),
                {"values": torch.randn(2, 6, 1, requires_grad=True)},
            ),
        ),
        (
            "covariate_embedding",
            lambda cls: (
                # Create config and attach expected attributes dynamically
                (lambda cfg: (setattr(cfg, "categorical_features", {"cat": 10}) or setattr(cfg, "numerical_features", 2) or cfg))(EmbeddingConfig(d_model=16, dropout=0.0)),
                {
                    "categorical_data": {"cat": torch.randint(0, 10, (2, 6))},
                    "numerical_data": torch.randn(2, 6, 2),
                },
            ),
        ),
        (
            "hybrid_embedding",
            lambda cls: (
                # Create config and attach flags; avoid passing extra dataclass fields
                (lambda cfg: (setattr(cfg, "use_temporal", True) or setattr(cfg, "use_value", True) or setattr(cfg, "use_covariate", True) or cfg))(EmbeddingConfig(d_model=16, dropout=0.0)),
                {
                    "values": torch.randn(2, 6, 1, requires_grad=True),
                    "temporal_features": {
                        "hour": torch.randint(0, 24, (2, 6)),
                        "day": torch.randint(1, 31, (2, 6)),
                        "weekday": torch.randint(0, 7, (2, 6)),
                        "month": torch.randint(1, 13, (2, 6)),
                    },
                    # numerical_data may be ignored if covariate sub-config has no numerical_features
                    "numerical_data": torch.randn(2, 6, 1),
                },
            ),
        ),
    ],
)
def test_embedding_forward_shape_and_grad(name: str, setup: Callable[[type], Tuple[Any, dict[str, Any]]]) -> None:
    """Instantiate embedding, run forward, check shape and gradient flows."""
    if get_global_registry is None or EmbeddingConfig is None or register_utils_embeddings is None:
        pytest.skip("Registry unavailable")
    register_utils_embeddings()
    reg = get_global_registry()
    comps = reg.list_components().get("embedding", [])
    if name not in comps:
        pytest.skip(f"{name} not registered")

    cls = reg.get("embedding", name)
    cfg_or_inst, kwargs = setup(cls)
    # setup may return a configured EmbeddingConfig or an already-instantiated module
    if isinstance(cfg_or_inst, EmbeddingConfig):
        emb = cls(cfg_or_inst)
    else:
        emb = cfg_or_inst

    # Identify a tensor input to backprop through
    grad_input = None
    for v in kwargs.values():
        if isinstance(v, torch.Tensor) and v.requires_grad:
            grad_input = v
            break
    if grad_input is None:
        # Try to mark one input as requires_grad to test gradients
        for k, v in list(kwargs.items()):
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                v.requires_grad_(True)
                grad_input = v
                kwargs[k] = v
                break

    out = emb(**kwargs)
    assert isinstance(out, torch.Tensor)
    # Expect [B, L, d_model]
    assert out.shape[0] == 2 and out.shape[1] == 6 and out.shape[2] == 16

    if grad_input is not None:
        out.mean().backward()
        assert grad_input.grad is not None and torch.isfinite(grad_input.grad).all()


def test_embedding_metadata_present() -> None:
    """At least one embedding reports metadata in the registry."""
    if get_global_registry is None or register_utils_embeddings is None:
        pytest.skip("Registry unavailable")
    register_utils_embeddings()
    reg = get_global_registry()
    meta_ok = 0
    for n in reg.list_components().get("embedding", []):
        try:
            md = reg.get_metadata("embedding", n)  # type: ignore[attr-defined]
            if isinstance(md, dict):
                meta_ok += 1
        except Exception:
            continue
    assert meta_ok >= 1
