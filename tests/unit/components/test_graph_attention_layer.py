import torch
import pytest

from models.Celestial_Enhanced_PGAT import GraphAttentionLayer


def test_graph_attention_per_batch_mask_shapes_and_forward():
    """Ensure per-batch adjacency masks are constructed and forward runs."""
    d_model = 16
    d_ff = 32
    n_heads = 4
    dropout = 0.0
    layer = GraphAttentionLayer(d_model, d_ff, n_heads, dropout)

    batch_size = 3
    seq_len = 5

    # Random inputs
    x = torch.randn(batch_size, seq_len, d_model)

    # Batch-specific adjacencies: identity, fully connected, ring
    eye = torch.eye(seq_len)
    full = torch.ones(seq_len, seq_len)
    ring = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        ring[i, i] = 1
        ring[i, (i + 1) % seq_len] = 1
        ring[i, (i - 1) % seq_len] = 1

    adj = torch.stack([eye, full, ring], dim=0)  # [B, L, L]

    # Forward should not error and preserve shape
    out = layer(x, adj)
    assert out.shape == x.shape


def test_graph_attention_mismatch_adj_size_fallback_identity():
    """If adjacency size mismatches seq_len, layer should fallback to identity and run."""
    d_model = 8
    d_ff = 16
    n_heads = 2
    dropout = 0.0
    layer = GraphAttentionLayer(d_model, d_ff, n_heads, dropout)

    batch_size = 2
    seq_len = 6
    x = torch.randn(batch_size, seq_len, d_model)

    # Provide smaller adjacency (will trigger identity fallback)
    adj = torch.eye(4)  # [4, 4] instead of [6, 6]

    out = layer(x, adj)
    assert out.shape == x.shape


def test_graph_attention_different_batch_adjacency_produces_different_outputs():
    """With same inputs but different adjacencies per batch, outputs should differ."""
    d_model = 12
    d_ff = 24
    n_heads = 3
    dropout = 0.0
    layer = GraphAttentionLayer(d_model, d_ff, n_heads, dropout)

    batch_size = 2
    seq_len = 4
    # Same input across batches
    x_single = torch.randn(1, seq_len, d_model)
    x = x_single.repeat(batch_size, 1, 1)

    # Batch 0: identity; Batch 1: fully connected
    eye = torch.eye(seq_len)
    full = torch.ones(seq_len, seq_len)
    adj = torch.stack([eye, full], dim=0)

    out = layer(x, adj)
    # Compare outputs for two batches; expect at least some difference
    diff = torch.abs(out[0] - out[1]).sum().item()
    assert diff > 1e-6