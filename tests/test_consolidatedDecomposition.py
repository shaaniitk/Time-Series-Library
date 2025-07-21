"""
Test for unified Decomposition module and registry
"""
import pytest
import torch
from utils.modular_components.implementations.Decomposition import DECOMPOSITION_REGISTRY, get_decomposition_method

def test_registry_keys():
    expected = {'series_decomp', 'stable_decomp', 'learnable_decomp', 'wavelet_decomp'}
    assert set(DECOMPOSITION_REGISTRY.keys()) == expected

def test_series_decomposition_instantiation():
    decomp = get_decomposition_method('series_decomp', kernel_size=5)
    assert decomp is not None
    assert hasattr(decomp, 'forward')
    # Test forward pass
    x = torch.randn(2, 10, 3)
    seasonal, trend = decomp(x)
    assert seasonal.shape == x.shape
    assert trend.shape == x.shape

def test_stable_decomposition_instantiation():
    decomp = get_decomposition_method('stable_decomp', kernel_size=5)
    assert decomp is not None
    x = torch.randn(2, 10, 3)
    seasonal, trend = decomp(x)
    assert seasonal.shape == x.shape
    assert trend.shape == x.shape

def test_learnable_decomposition_instantiation():
    decomp = get_decomposition_method('learnable_decomp', input_dim=3, init_kernel_size=5, max_kernel_size=10)
    assert decomp is not None
    x = torch.randn(2, 10, 3)
    seasonal, trend = decomp(x)
    assert seasonal.shape == x.shape
    assert trend.shape == x.shape

def test_wavelet_decomposition_instantiation():
    decomp = get_decomposition_method('wavelet_decomp', seq_len=10, d_model=3)
    assert decomp is not None
    # Forward not implemented, should raise NotImplementedError
    x = torch.randn(2, 10, 3)
    try:
        decomp(x)
    except NotImplementedError:
        pass
    else:
        assert False, "WaveletHierarchicalDecomposition forward should raise NotImplementedError"
