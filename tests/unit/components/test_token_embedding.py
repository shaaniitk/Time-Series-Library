import torch
import pytest

from models.Celestial_Enhanced_PGAT import TokenEmbedding


def test_token_embedding_valid_dimensions_forward():
    c_in = 7
    d_model = 16
    embed = TokenEmbedding(c_in=c_in, d_model=d_model)

    batch_size = 3
    seq_len = 10
    x = torch.randn(batch_size, seq_len, c_in)

    out = embed(x)
    assert out.shape == (batch_size, seq_len, d_model)


def test_token_embedding_raises_on_mismatched_input_dim():
    c_in = 5
    d_model = 8
    embed = TokenEmbedding(c_in=c_in, d_model=d_model)

    batch_size = 2
    seq_len = 6
    x = torch.randn(batch_size, seq_len, c_in + 1)  # mismatch

    with pytest.raises(ValueError) as exc:
        _ = embed(x)
    assert "does not match expected c_in" in str(exc.value)