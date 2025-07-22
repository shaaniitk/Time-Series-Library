"""
Holistic test for backbone components.
Ensures every backbone class can be instantiated and run a forward pass.
"""
import pytest
import torch
import inspect
from utils.modular_components.implementations.backbones import ChronosBackbone, T5Backbone
from utils.modular_components.implementations.simple_backbones import SimpleTransformerBackbone

BACKBONE_CLASSES = [ChronosBackbone, T5Backbone, SimpleTransformerBackbone]

@pytest.mark.parametrize("cls", BACKBONE_CLASSES)
def test_backbone_forward(cls):
    # Skip abstract classes
    if inspect.isabstract(cls):
        pytest.skip(f"Skipping abstract class: {cls.__name__}")
    # Create minimal config
    config = type('Config', (), {"d_model": 32, "dropout": 0.1, "model_name": "google/flan-t5-small", "pretrained": False})()
    backbone = cls(config)
    # Use correct input type for each backbone
    if cls in [ChronosBackbone, T5Backbone]:
        # T5/Chronos expects integer token IDs
        x = torch.randint(0, 100, (2, 8), dtype=torch.long)
        attention_mask = torch.ones(2, 8, dtype=torch.long)
        try:
            output = backbone(x, attention_mask)
        except Exception:
            output = backbone(x)
    else:
        # Simple backbone expects float tensor
        x = torch.randn(2, 8, 32)
        try:
            output = backbone(x)
        except Exception:
            output = backbone(x)
    assert output is not None
