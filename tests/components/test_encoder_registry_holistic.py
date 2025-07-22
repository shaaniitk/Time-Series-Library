"""
Holistic registry test for encoder components.
Ensures every registered encoder component can be instantiated and run a forward pass.
"""
import pytest
import torch
from utils.modular_components.implementations.encoder_unified import EncoderRegistry

@pytest.mark.parametrize("name", list(EncoderRegistry._components.keys()))
def test_encoder_registry_forward(name):
    component_class = EncoderRegistry._components[name]
    # Create minimal config or use default constructor
    try:
        config = type('Config', (), {"d_model": 32, "dropout": 0.1})()
        encoder = component_class(config)
    except Exception:
        encoder = component_class()
    # Create dummy input: [batch, seq_len, d_model]
    x = torch.randn(2, 8, 32)
    mask = torch.ones(2, 8).bool()
    # Forward pass (handle different signatures)
    try:
        output = encoder(x, mask)
    except Exception:
        output = encoder(x)
    assert output is not None
