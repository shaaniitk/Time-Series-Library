
import pytest
import torch
from layers.modular.decomposition.wavelet_decomposition import WaveletHierarchicalDecomposition

@pytest.fixture
def decomp_config():
    return {
        "seq_len": 96,
        "d_model": 512,
        "wavelet_type": "db4",
        "levels": 3
    }

def test_decomposition_instantiation(decomp_config):
    """Test that the component instantiates correctly."""
    decomp = WaveletHierarchicalDecomposition(**decomp_config)
    assert decomp is not None, "Decomposition should instantiate."

def test_decomposition_forward_pass(decomp_config):
    """Test the forward pass of the decomposition."""
    decomp = WaveletHierarchicalDecomposition(**decomp_config)
    x = torch.randn(2, decomp_config["seq_len"], decomp_config["d_model"])
    seasonal, trend = decomp(x)
    assert seasonal.shape == x.shape, "Seasonal component shape should match input shape."
    assert trend.shape == x.shape, "Trend component shape should match input shape."

if __name__ == "__main__":
    pytest.main()
