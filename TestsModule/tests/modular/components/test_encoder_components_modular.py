import torch
import torch.nn as nn
import pytest
from layers.modular.encoder.base_encoder import ModularEncoder
from layers.modular.encoder.standard_encoder import StandardEncoder
from layers.modular.encoder.enhanced_encoder import EnhancedEncoder
from layers.modular.encoder.stable_encoder import StableEncoder
from layers.modular.layers.standard_layers import StandardEncoderLayer
from layers.modular.attention.core.multihead_attention import MultiHeadAttention
from layers.modular.decomposition.series_decomposition import SeriesDecomposition as MovingAverageDecomposition

@pytest.fixture
def modular_encoder():
    """Returns a ModularEncoder instance for testing."""
    d_model = 32
    n_heads = 4
    d_ff = 64
    dropout = 0.1
    activation = "relu"
    num_encoder_layers = 2

    attention_comp = MultiHeadAttention(d_model, n_heads, dropout)
    decomp_comp = MovingAverageDecomposition(kernel_size=25)

    encoder_layers = [
        StandardEncoderLayer(
            attention_comp,
            decomp_comp,
            d_model,
            n_heads,
            d_ff,
            dropout=dropout,
            activation=activation
        ) for _ in range(num_encoder_layers)
    ]

    return ModularEncoder(encoder_layers)

@pytest.fixture
def standard_encoder():
    """Returns a StandardEncoder instance for testing."""
    d_model = 32
    n_heads = 4
    d_ff = 64
    dropout = 0.1
    activation = "relu"
    num_encoder_layers = 2

    attention_comp = MultiHeadAttention(d_model, n_heads, dropout)
    decomp_comp = MovingAverageDecomposition(kernel_size=25)

    return StandardEncoder(
        num_encoder_layers=num_encoder_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        attention_comp=attention_comp,
        decomp_comp=decomp_comp
    )

@pytest.fixture
def enhanced_encoder():
    """Returns an EnhancedEncoder instance for testing."""
    d_model = 32
    n_heads = 4
    d_ff = 64
    dropout = 0.1
    activation = "relu"
    num_encoder_layers = 2

    attention_comp = MultiHeadAttention(d_model, n_heads, dropout)
    decomp_comp = MovingAverageDecomposition(kernel_size=25)

    return EnhancedEncoder(
        num_encoder_layers=num_encoder_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        attention_comp=attention_comp,
        decomp_comp=decomp_comp
    )

@pytest.fixture
def stable_encoder():
    """Returns a StableEncoder instance for testing."""
    d_model = 32
    n_heads = 4
    d_ff = 64
    dropout = 0.1
    activation = "relu"
    num_encoder_layers = 2

    attention_comp = MultiHeadAttention(d_model, n_heads, dropout)
    decomp_comp = MovingAverageDecomposition(kernel_size=25)

    return StableEncoder(
        num_encoder_layers=num_encoder_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        attention_comp=attention_comp,
        decomp_comp=decomp_comp
    )

def test_modular_encoder_forward(modular_encoder):
    """Tests the forward pass of the ModularEncoder."""
    batch_size = 16
    seq_len = 100
    d_model = 32

    x = torch.randn(batch_size, seq_len, d_model)

    output, attns = modular_encoder(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert len(attns) == 2

def test_standard_encoder_forward(standard_encoder):
    """Tests the forward pass of the StandardEncoder."""
    batch_size = 16
    seq_len = 100
    d_model = 32

    x = torch.randn(batch_size, seq_len, d_model)

    output, attns = standard_encoder(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert len(attns) == 2

def test_enhanced_encoder_forward(enhanced_encoder):
    """Tests the forward pass of the EnhancedEncoder."""
    batch_size = 16
    seq_len = 100
    d_model = 32

    x = torch.randn(batch_size, seq_len, d_model)

    output, attns = enhanced_encoder(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert len(attns) == 2

def test_stable_encoder_forward(stable_encoder):
    """Tests the forward pass of the StableEncoder."""
    batch_size = 16
    seq_len = 100
    d_model = 32

    x = torch.randn(batch_size, seq_len, d_model)

    output, attns = stable_encoder(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert len(attns) == 2