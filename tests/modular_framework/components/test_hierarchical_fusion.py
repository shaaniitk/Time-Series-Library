
import pytest
import torch
from layers.modular.fusion.hierarchical_fusion import HierarchicalFusion

@pytest.fixture
def fusion_config():
    return {
        "d_model": 512,
        "n_levels": 3,
    }

def test_fusion_instantiation(fusion_config):
    """Test that the component instantiates correctly."""
    fusion = HierarchicalFusion(**fusion_config)
    assert fusion is not None, "Fusion should instantiate."

def test_fusion_forward_pass(fusion_config):
    """Test the forward pass of the fusion."""
    fusion = HierarchicalFusion(**fusion_config)
    x = [torch.randn(2, 96, 512), torch.randn(2, 48, 512), torch.randn(2, 24, 512)]
    fused_features = fusion(x)
    assert fused_features.shape == (2, 96, 512), "Fused features should have the correct shape."

if __name__ == "__main__":
    pytest.main()
