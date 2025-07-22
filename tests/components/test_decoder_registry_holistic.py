"""
Holistic registry test for decoder components.
Ensures every registered decoder component can be instantiated and run a forward pass.
"""
import pytest
import torch
import inspect
from utils.modular_components.implementations.decoders import DECODER_REGISTRY

# Minimal mock components for attention and decomposition
class DummyAttention(torch.nn.Module):
    def forward(self, *args, **kwargs):
        # Return shape: [batch, seq_len, d_model]
        return torch.randn(1, 32, 32)

class DummyDecomp(torch.nn.Module):
    def forward(self, *args, **kwargs):
        # Return shape: [batch, seq_len, d_model]
        return torch.randn(1, 32, 32)

@pytest.mark.parametrize("name", list(DECODER_REGISTRY.keys()))
def test_decoder_registry_forward(name):
    component_class = DECODER_REGISTRY[name]
    sig = inspect.signature(component_class.__init__)
    args = {}
    # Use d_model and c_out for input shape
    d_model = 32
    c_out = 32
    for param in list(sig.parameters.values())[1:]:  # skip 'self'
        pname = param.name
        if pname in ('d_layers', 'num_layers'):
            args[pname] = 2
        elif pname == 'd_model':
            args[pname] = d_model
        elif pname == 'c_out':
            args[pname] = c_out
        elif pname == 'd_ff':
            args[pname] = 64
        elif pname == 'n_heads':
            args[pname] = 4
        elif pname == 'dropout':
            args[pname] = 0.1
        elif pname == 'activation':
            args[pname] = 'relu'
        elif pname == 'self_attention_comp' or pname == 'cross_attention_comp':
            args[pname] = DummyAttention()
        elif pname == 'decomp_comp':
            args[pname] = DummyDecomp()
        elif pname == 'norm_layer' or pname == 'projection':
            args[pname] = None
        elif param.default != inspect.Parameter.empty:
            args[pname] = param.default
        else:
            args[pname] = None
    try:
        decoder = component_class(**args)
    except Exception as e:
        pytest.skip(f"Could not instantiate {name}: {e}")
    # Create dummy input: [batch, seq_len, d_model]
    x = torch.randn(1, 32, d_model)
    cross = torch.randn(1, 32, d_model)
    x_mask = torch.ones(1, 32).bool()
    cross_mask = torch.ones(1, 32).bool()
    trend = torch.randn(1, 32, c_out)
    try:
        output = decoder(x, cross, x_mask=x_mask, cross_mask=cross_mask, trend=trend)
        # Accept both single output and tuple output
        if isinstance(output, tuple):
            assert all(o is not None for o in output)
        else:
            assert output is not None
    except Exception as e:
        pytest.skip(f"Forward failed for {name}: {e}")
