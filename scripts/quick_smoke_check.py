import torch

from unified_component_registry import unified_registry

# Import configs for components
from utils.implementations.attention.restored_algorithms import RestoredFourierConfig
from utils.implementations.attention.layers_wrapped_attentions import (
    WaveletAttentionConfig,
    BayesianAttentionConfig,
)


def run_smoke():
    print("Listing components from unified registry...")
    comps = unified_registry.list_all_components()
    print({k: len(v) for k, v in comps.items()})

    # Restored Fourier Attention
    f_cfg = RestoredFourierConfig(d_model=64, num_heads=4, dropout=0.1, seq_len=32)
    f_attn = unified_registry.create_component('attention', 'restored_fourier_attention', f_cfg)
    x = torch.randn(2, 32, 64)
    y, attn = f_attn.apply_attention(x, x, x)
    assert y.shape == x.shape
    print("RestoredFourierAttention OK", y.shape)

    # Legacy Wavelet (wrapped)
    w_cfg = WaveletAttentionConfig(d_model=64, num_heads=4, dropout=0.1, levels=2, n_levels=2)
    w_attn = unified_registry.create_component('attention', 'layers_wavelet_attention', w_cfg)
    x = torch.randn(2, 16, 64)
    y, attn = w_attn.apply_attention(x, x, x)
    assert y.shape == x.shape
    print("Wrapped WaveletAttention OK", y.shape)

    # Legacy Bayesian (wrapped)
    b_cfg = BayesianAttentionConfig(d_model=64, num_heads=4, dropout=0.1, prior_std=0.5, temperature=0.8)
    b_attn = unified_registry.create_component('attention', 'layers_bayesian_attention', b_cfg)
    x = torch.randn(2, 8, 64)
    y, attn = b_attn.apply_attention(x, x, x)
    assert y.shape == x.shape
    print("Wrapped BayesianAttention OK", y.shape)

    print("\nSMOKE CHECK: SUCCESS")


if __name__ == "__main__":
    run_smoke()
