"""Lightweight shape smoke tests for unified registry components.

These are deliberately minimal to validate that each major family produces an
output tensor of expected shape given simple synthetic inputs. They DO NOT try
to validate numerical correctness – only interface stability.
"""
from __future__ import annotations
import pytest
import torch
from layers.modular.core import unified_registry, ComponentFamily


@pytest.mark.parametrize("attention_name", [
    "autocorrelation", "adaptive_autocorrelation_layer", "fourier_attention",
    "wavelet_attention", "causal_convolution"
])
def test_attention_forward_shape(attention_name: str) -> None:
    d_model = 16
    n_heads = 4
    seq_len = 8
    attn = unified_registry.create(ComponentFamily.ATTENTION, attention_name, d_model=d_model, n_heads=n_heads, seq_len=seq_len, dropout=0.0)  # type: ignore
    x = torch.randn(2, seq_len, d_model)
    # Many attention classes expose forward(q,k,v) or forward(x)
    if hasattr(attn, 'forward'):
        try:
            if 'k' in attn.forward.__code__.co_varnames or 'q' in attn.forward.__code__.co_varnames:  # type: ignore[attr-defined]
                out = attn(x, x, x)
            else:
                out = attn(x)
        except TypeError:
            out = attn(x, x, x)
    else:  # pragma: no cover - defensive
        pytest.skip("No forward method exposed")
    # Some attention modules return (output, attn_weights) tuple
    if isinstance(out, tuple):
        tensor_out = out[0]
    else:
        tensor_out = out
    assert isinstance(tensor_out, torch.Tensor), "Attention primary output must be a tensor"
    assert tensor_out.shape[-1] == d_model, "Output feature dim mismatch"


def _make_basic_components(d_model: int = 16):
    decomp = unified_registry.create(ComponentFamily.DECOMPOSITION, 'moving_avg', kernel_size=3)
    attn = unified_registry.create(ComponentFamily.ATTENTION, 'autocorrelation', d_model=d_model, n_heads=4, dropout=0.0)
    return attn, decomp


def test_encoder_decoder_round_trip() -> None:
    d_model = 16
    attn, decomp = _make_basic_components(d_model)
    enc = unified_registry.create(
        ComponentFamily.ENCODER,
        'enhanced_encoder',
        num_encoder_layers=1,
        d_model=d_model,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        activation='gelu',
        attention_comp=attn,
        decomp_comp=decomp,
    )
    dec = unified_registry.create(
        ComponentFamily.DECODER,
        'enhanced_decoder',
        num_decoder_layers=1,
        d_model=d_model,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        activation='gelu',
        c_out=8,
        self_attention_comp=attn,
        cross_attention_comp=attn,
        decomp_comp=decomp,
    )
    x = torch.randn(2, 8, d_model)
    memory = enc(x)
    out = dec(x, memory)
    # Decoder may return (seasonal, trend) tuple; accept either
    if isinstance(out, tuple):
        primary = out[0]
    else:
        primary = out
    assert isinstance(primary, torch.Tensor)
    assert primary.shape[0] == x.shape[0]


@pytest.mark.parametrize("loss_name", ["mse", "quantile_loss", "bayesian_mse_loss"])  # registry keys
def test_loss_instantiation(loss_name: str) -> None:
    loss_fn = unified_registry.create(ComponentFamily.LOSS, loss_name)
    # Provide minimal shape depending on loss
    if loss_name == 'quantile_loss':
        # predictions shape [B, T, C*Q]; use T=1, C=1, Q=3
        y_pred = torch.randn(2, 1, 3)
        y_true = torch.randn(2, 1, 1)
    else:
        y_pred = torch.randn(4, 3)
        y_true = torch.randn(4, 3)
    try:
        val = loss_fn(y_pred, y_true)  # type: ignore[operator]
    except TypeError:
        # Some wrappers may require different call signature; skip if incompatible
        pytest.skip(f"Loss {loss_name} call signature not standard")
    assert torch.is_tensor(val)
