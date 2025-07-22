"""
Holistic registry test for embedding components.
Ensures every registered embedding component can be instantiated and run a forward pass.
"""
import pytest
import torch
from utils.modular_components.implementations.embeddings import EMBEDDING_REGISTRY

@pytest.mark.parametrize("name", list(EMBEDDING_REGISTRY.keys()))
def test_embedding_registry_forward(name):
    component_class = EMBEDDING_REGISTRY[name]
    # Create minimal config or use default constructor
    try:
        config = type('Config', (), {"d_model": 32, "dropout": 0.1})()
        embedding = component_class(config)
    except Exception:
        embedding = component_class()
    # Create dummy input: [batch, seq_len, d_model]
    x = torch.randn(2, 8, 32)
    # Forward pass (handle different signatures)
    try:
        output = embedding(x)
    except Exception:
        output = embedding(x)
    assert output is not None
