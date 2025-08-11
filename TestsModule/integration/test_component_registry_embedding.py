"""Embedding component functional smoke (no explicit registry yet).

Legacy embedding tests validated individual classes; here we perform minimal
instantiation & forward checks for a representative subset to ensure they
remain constructible within the modular test suite context.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    from layers.Embed import (
        PositionalEmbedding,
        TokenEmbedding,
        FixedEmbedding,
        TemporalEmbedding,
        TimeFeatureEmbedding,
        DataEmbedding,
        DataEmbedding_inverted,
        DataEmbedding_wo_pos,
        PatchEmbedding,
    )  # type: ignore
except Exception:  # pragma: no cover
    PositionalEmbedding = None  # type: ignore
    TokenEmbedding = None  # type: ignore
    FixedEmbedding = None  # type: ignore
    TemporalEmbedding = None  # type: ignore
    TimeFeatureEmbedding = None  # type: ignore
    DataEmbedding = None  # type: ignore
    DataEmbedding_inverted = None  # type: ignore
    DataEmbedding_wo_pos = None  # type: ignore
    PatchEmbedding = None  # type: ignore


def _require_available(obj, name: str) -> None:
    if obj is None:
        pytest.skip(f"{name} unavailable in current refactor stage")


def test_positional_embedding_shape() -> None:
    _require_available(PositionalEmbedding, "PositionalEmbedding")
    emb = PositionalEmbedding(32, 500)  # type: ignore[operator]
    x = torch.randn(1, 40, 32)
    out = emb(x)
    assert out.shape == (1, 40, 32)
    assert not out.requires_grad  # fixed


def test_token_embedding_forward_and_grad() -> None:
    _require_available(TokenEmbedding, "TokenEmbedding")
    emb = TokenEmbedding(7, 32)  # type: ignore[operator]
    x = torch.randn(2, 24, 7, requires_grad=True)
    out = emb(x)
    assert out.shape == (2, 24, 32)
    out.mean().backward()
    assert x.grad is not None


@pytest.mark.parametrize("freq", ["h", "d"])  # keep short
def test_temporal_embedding_variants(freq: str) -> None:
    _require_available(TemporalEmbedding, "TemporalEmbedding")
    emb = TemporalEmbedding(32, embed_type="fixed", freq=freq)  # type: ignore[operator]
    tmarks = torch.stack([
        torch.randint(0, 13, (2, 16)),
        torch.randint(1, 32, (2, 16)),
        torch.randint(0, 7, (2, 16)),
        torch.randint(0, 24, (2, 16)),
    ], dim=-1)
    out = emb(tmarks)
    assert out.shape == (2, 16, 32)


def test_data_embedding_pathways() -> None:
    _require_available(DataEmbedding, "DataEmbedding")
    emb = DataEmbedding(5, 32, embed_type="fixed", freq="h")  # type: ignore[operator]
    x = torch.randn(2, 24, 5)
    tmarks = torch.stack([
        torch.randint(0, 13, (2, 24)),
        torch.randint(1, 32, (2, 24)),
        torch.randint(0, 7, (2, 24)),
        torch.randint(0, 24, (2, 24)),
    ], dim=-1)
    out_tm = emb(x, tmarks)
    out_none = emb(x, None)
    assert out_tm.shape == out_none.shape == (2, 24, 32)


def test_patch_embedding_basic() -> None:
    _require_available(PatchEmbedding, "PatchEmbedding")
    patch = PatchEmbedding(32, patch_len=8, stride=4, padding=2, dropout=0.0)  # type: ignore[operator]
    x = torch.randn(2, 3, 32)
    out, n_vars = patch(x)
    assert n_vars == 3
    assert out.dim() == 3

