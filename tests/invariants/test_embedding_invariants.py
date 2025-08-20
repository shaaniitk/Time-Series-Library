"""Embedding & positional invariants (Phase 3 - Embeddings).

Implemented now: pure positional sinusoidal embedding tests (temporal feature tests postponed until thresholds defined).

Covered:
1. Value bounds within [-1,1].
2. Per-position norm variance small.
3. Slice consistency between long sequence prefix and standalone shorter sequence.
4. Positional diversity (average absolute cosine similarity of random pairs not excessively high).
"""
from __future__ import annotations

import pytest
import torch

from .thresholds import get_threshold
from . import metrics as M


def _get_positional_embedding():  # type: ignore[return-type]
    try:
        from layers.Embed import PositionalEmbedding  # type: ignore
    except Exception:  # pragma: no cover
        pytest.skip("PositionalEmbedding unavailable")
    return PositionalEmbedding(d_model=32, max_len=512)


@pytest.mark.invariant
def test_positional_value_bounds():
    emb = _get_positional_embedding()
    dummy = torch.zeros(1, 256, 32)
    with torch.no_grad():
        pe = emb(dummy)
    assert pe.abs().max().item() <= get_threshold("embedding_positional_value_abs_max")


@pytest.mark.invariant
def test_positional_norm_stability():
    emb = _get_positional_embedding()
    dummy = torch.zeros(1, 256, 32)
    with torch.no_grad():
        pe = emb(dummy).squeeze(0)
    norms = torch.linalg.norm(pe, dim=-1)
    var = norms.var(unbiased=False).item()
    assert var <= get_threshold("embedding_positional_norm_var_max"), var


@pytest.mark.invariant
def test_positional_slice_consistency():
    emb = _get_positional_embedding()
    L = 200
    k = 123
    base = torch.zeros(1, L, 32)
    base_short = torch.zeros(1, k, 32)
    with torch.no_grad():
        full = emb(base)[:, :k]
        short = emb(base_short)
    rel_err = M.l2_relative_error(full, short)
    assert rel_err < get_threshold("embedding_slice_consistency_rel_err"), rel_err


@pytest.mark.invariant
def test_positional_diversity_cosine_mean_abs():
    emb = _get_positional_embedding()
    L = 128
    base = torch.zeros(1, L, 32)
    with torch.no_grad():
        pe = emb(base).squeeze(0)
    idx = torch.randint(0, L, (64,))
    jdx = torch.randint(0, L, (64,))
    mask = idx != jdx
    idx, jdx = idx[mask], jdx[mask]
    a = pe[idx]
    b = pe[jdx]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).abs().mean().item()
    assert cos <= get_threshold("embedding_positional_offdiag_mean_abs_cos_max"), cos


__all__ = []
