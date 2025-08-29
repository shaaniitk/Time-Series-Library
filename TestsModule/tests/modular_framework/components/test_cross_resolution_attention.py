
import pytest
import torch
from layers.modular.attention.cross_resolution_attention import CrossResolutionAttention

@pytest.fixture
def attention_config():
    return {
        "d_model": 512,
        "n_levels": 3,
        "n_heads": 8,
    }

def test_attention_instantiation(attention_config):
    """Test that the component instantiates correctly."""
    attention = CrossResolutionAttention(**attention_config)
    assert attention is not None, "Attention should instantiate."

def test_attention_forward_pass(attention_config):
    """Test the forward pass of the attention."""
    attention = CrossResolutionAttention(**attention_config)
    x = [torch.randn(2, 96, 512), torch.randn(2, 48, 512), torch.randn(2, 24, 512)]
    attended_features, _ = attention(x)
    assert len(attended_features) == len(x), "Should return the same number of levels."
    for i in range(len(x)):
        assert attended_features[i].shape == x[i].shape, f"Shape mismatch at level {i}."

if __name__ == "__main__":
    pytest.main()
