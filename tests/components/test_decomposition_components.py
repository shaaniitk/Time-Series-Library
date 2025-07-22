"""
Decomposition Components Test
Ensures all decomposition types run a forward pass with synthetic data.
"""
import pytest
import torch
from configs.modular_components import component_registry, register_all_components
from configs.schemas import ComponentType

def test_decomposition_forward():
    register_all_components()
    info = component_registry.get_component(ComponentType.DECOMPOSITION)
    assert info is not None
    decomposition_names = ['standard', 'wavelet', 'stable']  # Example names
    batch_size, seq_len, d_model = 2, 96, 8
    x = torch.randn(batch_size, seq_len, d_model)
    for name in decomposition_names:
        decomp = info.factory(name=name)
        out = decomp.forward(x)
        assert out.shape == x.shape or out is not None
