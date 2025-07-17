
import pytest
import torch
from layers.modular.encoder.hierarchical_encoder import HierarchicalEncoder
from argparse import Namespace

@pytest.fixture
def encoder_config():
    configs = Namespace(
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        activation='gelu',
        factor=1,
    )
    return {
        "configs": configs,
        "n_levels": 3,
    }

def test_encoder_instantiation(encoder_config):
    """Test that the component instantiates correctly."""
    encoder = HierarchicalEncoder(**encoder_config)
    assert encoder is not None, "Encoder should instantiate."

def test_encoder_forward_pass(encoder_config):
    """Test the forward pass of the encoder."""
    encoder = HierarchicalEncoder(**encoder_config)
    x = [torch.randn(2, 96, 512), torch.randn(2, 48, 512), torch.randn(2, 24, 512)]
    encoded_features, _ = encoder(x)
    assert len(encoded_features) == len(x), "Should return the same number of levels."
    for i in range(len(x)):
        assert encoded_features[i].shape == x[i].shape, f"Shape mismatch at level {i}."

if __name__ == "__main__":
    pytest.main()
