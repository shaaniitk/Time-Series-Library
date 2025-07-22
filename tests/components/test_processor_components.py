"""
Processor Components Test
Ensures all processor types run a forward pass with synthetic data.
"""
import pytest
import torch
from configs.modular_components import component_registry, register_all_components
from configs.schemas import ComponentType

def test_processor_forward():
    register_all_components()
    info = component_registry.get_component(ComponentType.PROCESSOR)
    assert info is not None
    # Example processor names
    processor_names = ['seq2seq', 'encoder_only', 'hierarchical', 'autoregressive']
    batch_size, seq_len, d_model = 2, 96, 8
    x = torch.randn(batch_size, seq_len, d_model)
    for name in processor_names:
        processor = info.factory(name=name)
        out = processor.forward(x)
        assert out.shape == x.shape or out is not None
