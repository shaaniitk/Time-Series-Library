"""
Holistic registry test for attention components.
Ensures every registered attention component can be instantiated and run a forward pass.
"""

import pytest
import torch
import inspect
from utils.modular_components.implementations.Attention import ATTENTION_REGISTRY

@pytest.mark.parametrize("name", list(ATTENTION_REGISTRY.keys()))
def test_attention_registry_forward(name):
    component_class = ATTENTION_REGISTRY[name]
    # Skip abstract classes
    if inspect.isabstract(component_class):
        pytest.skip(f"Skipping abstract class: {name}")
    # Create minimal config or use default constructor
    try:
        config = type('Config', (), {"d_model": 32, "dropout": 0.1})()
        attention = component_class(config)
    except Exception:
        attention = component_class()
    # Create dummy input: [batch, seq_len, d_model]
    x = torch.randn(2, 8, 32)
    mask = torch.ones(2, 8).bool()
    # Forward pass (handle different signatures)
    try:
        output = attention(x, mask)
    except Exception:
        output = attention(x)
    assert output is not None
