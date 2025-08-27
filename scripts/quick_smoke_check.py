import torch

from tools.unified_component_registry import unified_registry
from layers.modular.core import unified_registry as _ur  # ensure modular registrations
import layers.modular.core.register_components  # noqa: F401


def run_smoke():
    print("Listing components from unified registry...")
    comps = unified_registry.list_all_components()
    print({k: len(v) for k, v in comps.items()})

    # Fourier Attention (modular)
    f_attn = _ur.create_component('attention', 'fourier_attention', d_model=64, n_heads=4, dropout=0.1, seq_len=32)
    x = torch.randn(2, 32, 64)
    y, attn = f_attn(x, x, x)
    assert y.shape == x.shape
    print("FourierAttention OK", y.shape)

    # Wavelet Attention (modular)
    w_attn = _ur.create_component('attention', 'wavelet_attention', d_model=64, n_heads=4, dropout=0.1)
    x = torch.randn(2, 16, 64)
    y, attn = w_attn(x, x, x)
    assert y.shape == x.shape
    print("WaveletAttention OK", y.shape)

    # Bayesian Attention (modular)
    b_attn = _ur.create_component('attention', 'bayesian_attention', d_model=64, n_heads=4, dropout=0.1)
    x = torch.randn(2, 8, 64)
    y, attn = b_attn(x, x, x)
    assert y.shape == x.shape
    print("BayesianAttention OK", y.shape)

    print("\nSMOKE CHECK: SUCCESS")


if __name__ == "__main__":
    run_smoke()
