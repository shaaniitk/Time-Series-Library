"""Utility factory helpers for modular component tests.

Centralizes minimal constructor argument logic so individual tests can stay
focused on behavioural assertions instead of boilerplate setup.
"""
from __future__ import annotations

from typing import Any, Dict

import torch

try:  # pragma: no cover - optional imports during partial refactors
    from layers.modular.core import get_attention_component  # type: ignore
    import layers.modular.core.register_components  # noqa: F401  # populate registry side-effects
    from layers.modular.decomposition import get_decomposition_component  # type: ignore
    from layers.modular.encoder import get_encoder_component  # type: ignore
    from layers.modular.decoder import get_decoder_component  # type: ignore
    from layers.modular.sampling import get_sampling_component  # type: ignore
    from layers.modular.output_heads import get_output_head_component  # type: ignore
except Exception:  # pragma: no cover
    get_attention_component = None  # type: ignore
    get_decomposition_component = None  # type: ignore
    get_encoder_component = None  # type: ignore
    get_decoder_component = None  # type: ignore
    get_sampling_component = None  # type: ignore
    get_output_head_component = None  # type: ignore


def make_decomposition(name: str, d_model: int, seq_len: int) -> Any:
    """Instantiate a decomposition component with minimal safe kwargs."""
    if get_decomposition_component is None:  # pragma: no cover
        raise RuntimeError("Decomposition registry unavailable")
    if name in {"series_decomp", "stable_decomp"}:
        return get_decomposition_component(name, kernel_size=5)
    if name == "learnable_decomp":
        return get_decomposition_component(name, input_dim=d_model, init_kernel_size=5, max_kernel_size=8)
    if name == "wavelet_decomp":
        return get_decomposition_component(name, seq_len=seq_len, d_model=d_model, levels=2)
    return get_decomposition_component(name)


def make_attention(name: str, d_model: int, n_heads: int, seq_len: int) -> Any:
    """Instantiate an attention component with adaptive parameter handling."""
    if get_attention_component is None:  # pragma: no cover
        raise RuntimeError("Attention registry unavailable")
    params: Dict[str, Any] = {"d_model": d_model, "n_heads": n_heads, "dropout": 0.0}
    if name in {"autocorrelation_layer", "adaptive_autocorrelation_layer", "enhanced_autocorrelation", "new_adaptive_autocorrelation_layer"}:
        params["factor"] = 1
    elif name == "cross_resolution_attention":
        params["n_levels"] = 2
    elif name == "fourier_attention":
        params["seq_len"] = seq_len
    elif name == "fourier_block":
        params = {"in_channels": d_model, "out_channels": d_model, "seq_len": seq_len}
    elif name == "fourier_cross_attention":
        params = {"in_channels": d_model, "out_channels": d_model, "seq_len_q": seq_len, "seq_len_kv": seq_len}
    elif name in {"wavelet_attention", "wavelet_decomposition"}:
        params["n_levels"] = 2
    elif name == "adaptive_wavelet_attention":
        params["max_levels"] = 3
    elif name == "multi_scale_wavelet_attention":
        params["scales"] = [1, 2]
    elif name == "hierarchical_autocorrelation":
        params.update({"hierarchy_levels": [1, 4], "factor": 1})
    elif name in {"bayesian_attention", "bayesian_cross_attention"}:
        params["prior_std"] = 1.0
    elif name == "bayesian_multi_head_attention":
        params.update({"prior_std": 1.0, "n_samples": 2})
    elif name == "meta_learning_adapter":
        params.update({"adaptation_steps": 1, "meta_lr": 0.01, "inner_lr": 0.1})
    elif name == "adaptive_mixture":
        params["mixture_components"] = 2
    elif name == "causal_convolution":
        params.update({"kernel_sizes": [3], "dilation_rates": [1]})
    elif name == "temporal_conv_net":
        params.update({"num_levels": 2, "kernel_size": 3})
    elif name == "convolutional_attention":
        params.update({"conv_kernel_size": 3, "pool_size": 2})
    return get_attention_component(name, **params)


def make_encoder(name: str, *, d_model: int, n_heads: int) -> Any:
    if get_encoder_component is None:  # pragma: no cover
        raise RuntimeError("Encoder registry unavailable")
    attention = make_attention("autocorrelation_layer", d_model, n_heads, seq_len=16)
    decomp = make_decomposition("series_decomp", d_model, seq_len=16)
    if name == "hierarchical":
        return get_encoder_component(
            name,
            e_layers=1,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=2 * d_model,
            dropout=0.0,
            activation="gelu",
            attention_type="autocorrelation_layer",
            decomp_type="series_decomp",
            decomp_params={"kernel_size": 5},
            n_levels=2,
            share_weights=False,
        )
    return get_encoder_component(
        name,
        e_layers=1,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=2 * d_model,
        dropout=0.0,
        activation="gelu",
        attention_comp=attention,
        decomp_comp=decomp,
    )


def make_decoder(name: str, *, d_model: int, n_heads: int, c_out: int) -> Any:
    if get_decoder_component is None:  # pragma: no cover
        raise RuntimeError("Decoder registry unavailable")
    attention = make_attention("autocorrelation_layer", d_model, n_heads, seq_len=16)
    decomp = make_decomposition("learnable_decomp", d_model, seq_len=16)
    return get_decoder_component(
        name,
        d_layers=1,
        d_model=d_model,
        c_out=c_out,
        n_heads=n_heads,
        d_ff=2 * d_model,
        dropout=0.0,
        activation="gelu",
        self_attention_comp=attention,
        cross_attention_comp=attention,
        decomp_comp=decomp,
    )


def make_sampling(name: str, **kwargs: Any) -> Any:
    if get_sampling_component is None:  # pragma: no cover
        raise RuntimeError("Sampling registry unavailable")
    return get_sampling_component(name, **kwargs)


def make_output_head(name: str, d_model: int, c_out: int, **kwargs: Any) -> Any:
    if get_output_head_component is None:  # pragma: no cover
        raise RuntimeError("Output head registry unavailable")
    return get_output_head_component(name, d_model=d_model, c_out=c_out, **kwargs)


def random_series(batch: int, length: int, d_model: int, requires_grad: bool = False) -> torch.Tensor:
    return torch.randn(batch, length, d_model, requires_grad=requires_grad)
