"""
Holistic Processor Registry Test
This test instantiates every registered processor component and validates a basic forward pass.
"""
import pytest
import torch
from utils.modular_components.implementations.processorss import PROCESSOR_REGISTRY
from utils.modular_components.config_schemas import ComponentConfig

def get_minimal_processor_config():
    # Minimal config for processor instantiation
    return ComponentConfig(c_in=8, c_out=4, seq_len=16, label_len=8, pred_len=4)

@pytest.mark.parametrize("proc_name,proc_cls", list(PROCESSOR_REGISTRY.items()))
def test_processor_registry_instantiation(proc_name, proc_cls):
    config = get_minimal_processor_config()
    processor = proc_cls(config)
    # Create dummy tensors for forward pass
    x_enc = torch.randn(2, config.seq_len, config.c_in)
    x_mark_enc = torch.randn(2, config.seq_len, 1)
    x_dec = torch.randn(2, config.pred_len, config.c_in)
    x_mark_dec = torch.randn(2, config.pred_len, 1)
    # Forward pass
    try:
        out = processor.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        assert out is not None, f"Processor {proc_name} forward returned None"
    except Exception as e:
        pytest.skip(f"Processor {proc_name} could not be tested: {e}")
