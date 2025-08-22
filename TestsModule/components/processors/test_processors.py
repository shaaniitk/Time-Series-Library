from __future__ import annotations

"""Processor component tests.

Covers utils-wrapped processors for decomposition, encoder, decoder, and fusion.
Also verifies specialized processors register successfully. Lightweight and
marked extended. Uses tiny tensors and checks forward shape + grad flow where
appropriate.
"""

from typing import List

import pytest
import torch

pytestmark = [pytest.mark.extended]

try:  # pragma: no cover
    # Global registry and registration hooks
    from utils.modular_components.registry import get_global_registry  # type: ignore
    from utils.implementations.decomposition.wrapped_decompositions import (  # type: ignore
        register_layers_decompositions,
        DecompositionProcessorConfig,
    )
    from utils.implementations.encoder.wrapped_encoders import (  # type: ignore
        register_layers_encoders,
        EncoderProcessorConfig,
    )
    from utils.implementations.decoder.wrapped_decoders import (  # type: ignore
        register_layers_decoders,
        DecoderProcessorConfig,
    )
    from utils.implementations.fusion.wrapped_fusions import (  # type: ignore
        register_layers_fusions,
        FusionProcessorConfig,
    )
    from utils.modular_components.implementations.register_advanced import (  # type: ignore
        register_specialized_processors,
    )
    # Legacy components needed as dependencies for encoder/decoder wrappers
    from layers.modular.core import get_attention_component  # type: ignore
    import layers.modular.core.register_components  # noqa: F401  # populate registry side-effects
    from layers.modular.decomposition import get_decomposition_component  # type: ignore
except Exception:  # pragma: no cover
    get_global_registry = None  # type: ignore
    register_layers_decompositions = None  # type: ignore
    DecompositionProcessorConfig = None  # type: ignore
    register_layers_encoders = None  # type: ignore
    EncoderProcessorConfig = None  # type: ignore
    register_layers_decoders = None  # type: ignore
    DecoderProcessorConfig = None  # type: ignore
    register_layers_fusions = None  # type: ignore
    FusionProcessorConfig = None  # type: ignore
    register_specialized_processors = None  # type: ignore
    get_attention_component = None  # type: ignore
    get_decomposition_component = None  # type: ignore


def _ensure_registered() -> None:
    """Register wrapped processors into the global registry if available."""
    if (
        get_global_registry is None
        or register_layers_decompositions is None
        or register_layers_encoders is None
        or register_layers_decoders is None
        or register_layers_fusions is None
    ):
        pytest.skip("Processor registry unavailable")
    register_layers_decompositions()
    register_layers_encoders()
    register_layers_decoders()
    register_layers_fusions()


def _has(name: str) -> bool:
    if get_global_registry is None:
        return False
    reg = get_global_registry()
    return name in reg.list_components().get("processor", [])


def test_processor_registry_non_empty() -> None:
    """At least one wrapped processor is registered and names are unique."""
    _ensure_registered()
    reg = get_global_registry()
    names = reg.list_components().get("processor", [])
    assert isinstance(names, list) and names
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "name",
    [
        "series_decomposition_processor",
        "stable_series_decomposition_processor",
        "learnable_series_decomposition_processor",
        "wavelet_hierarchical_decomposition_processor",
    ],
)
def test_decomposition_processors_forward_and_grad(name: str) -> None:
    """Wrapped decomposition processors return [B, L, D] and support grads."""
    _ensure_registered()
    if not _has(name):
        pytest.skip(f"{name} not registered")

    reg = get_global_registry()
    cls = reg.get("processor", name)
    cfg = DecompositionProcessorConfig(d_model=16, seq_len=8, kernel_size=5)
    proc = cls(cfg)

    x = torch.randn(2, 8, 16, requires_grad=True)
    out = proc(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "name",
    [
        "encoder_standard_processor",
        "encoder_enhanced_processor",
        "encoder_stable_processor",
        "encoder_hierarchical_processor",
    ],
)
def test_encoder_processors_forward_and_grad(name: str) -> None:
    """Wrapped encoder processors process [B, L, D] -> [B, L, D] with grads."""
    _ensure_registered()
    if not _has(name):
        pytest.skip(f"{name} not registered")
    if get_attention_component is None or get_decomposition_component is None:
        pytest.skip("Legacy attention/decomposition unavailable")

    reg = get_global_registry()
    cls = reg.get("processor", name)
    # Build tiny deps
    attn = get_attention_component("autocorrelation_layer", d_model=16, n_heads=2, dropout=0.0)
    decomp = get_decomposition_component("series_decomp", kernel_size=3)

    if name == "encoder_hierarchical_processor":
        cfg = EncoderProcessorConfig(
            d_model=16,
            dropout=0.0,
            n_levels=2,
            custom_params={
                "e_layers": 1,
                "n_heads": 2,
                "d_ff": 64,
                "activation": "gelu",
                "attention_type": "autocorrelation_layer",
                "decomp_type": "series_decomp",
                "decomp_params": {"kernel_size": 3},
            },
        )
    else:
        cfg = EncoderProcessorConfig(
            d_model=16,
            dropout=0.0,
            custom_params={
                "e_layers": 1,
                "d_ff": 64,
                "activation": "gelu",
                "attention_component": attn,
                "decomposition_component": decomp,
            },
        )
    proc = cls(cfg)

    x = torch.randn(2, 8, 16, requires_grad=True)
    out = proc(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
    out.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "name",
    [
        "decoder_standard_processor",
        "decoder_enhanced_processor",
        "decoder_stable_processor",
    ],
)
def test_decoder_processors_forward_and_grad(name: str) -> None:
    """Wrapped decoder processors map (tgt, enc_out, trend)->[B, L, c_out] with grads."""
    _ensure_registered()
    if not _has(name):
        pytest.skip(f"{name} not registered")
    if get_attention_component is None or get_decomposition_component is None:
        pytest.skip("Legacy attention/decomposition unavailable")

    reg = get_global_registry()
    cls = reg.get("processor", name)
    self_attn = get_attention_component("autocorrelation_layer", d_model=16, n_heads=2, dropout=0.0)
    decomp = get_decomposition_component("learnable_decomp", input_dim=16, init_kernel_size=3, max_kernel_size=6)

    cfg = DecoderProcessorConfig(
        d_model=16,
        c_out=16,
        dropout=0.0,
        custom_params={
            "d_layers": 1,
            "d_ff": 64,
            "activation": "gelu",
            "self_attention_component": self_attn,
            "cross_attention_component": self_attn,
            "decomposition_component": decomp,
            "c_out": 16,
        },
    )
    proc = cls(cfg)

    enc_out = torch.randn(2, 8, 16)
    tgt = torch.randn(2, 6, 16, requires_grad=True)
    trend = torch.zeros_like(tgt)
    out = proc(tgt, enc_out, trend=trend)
    assert isinstance(out, torch.Tensor)
    assert out.shape == tgt.shape
    out.sum().backward()
    assert tgt.grad is not None and torch.isfinite(tgt.grad).all()


@pytest.mark.parametrize("strategy", ["weighted_concat", "weighted_sum", "attention_fusion"])
def test_fusion_hierarchical_processor_forward_and_grad(strategy: str) -> None:
    """Hierarchical fusion processes multi-resolution features to a target length across strategies."""
    _ensure_registered()
    if not _has("fusion_hierarchical_processor"):
        pytest.skip("fusion_hierarchical_processor not registered")

    reg = get_global_registry()
    cls = reg.get("processor", "fusion_hierarchical_processor")
    cfg = FusionProcessorConfig(d_model=16, n_levels=3, fusion_strategy=strategy)
    proc = cls(cfg)

    # Create synthetic multi-resolution features
    base = torch.randn(2, 8, 16, requires_grad=True)
    lvl2 = torch.nn.functional.avg_pool1d(base.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
    lvl3 = torch.nn.functional.avg_pool1d(base.transpose(1, 2), kernel_size=4, stride=4).transpose(1, 2)
    feats: List[torch.Tensor] = [base, lvl2.detach(), lvl3.detach()]  # only base requires grad

    out = proc(feats, target_length=8)
    assert isinstance(out, torch.Tensor)
    assert out.shape == base.shape
    out.mean().backward()
    assert base.grad is not None and torch.isfinite(base.grad).all()


def test_specialized_processors_are_registered() -> None:
    """Specialized processors should register names in the global registry."""
    if register_specialized_processors is None or get_global_registry is None:
        pytest.skip("Registry unavailable for specialized processors")
    register_specialized_processors()
    reg = get_global_registry()
    names = reg.list_components().get("processor", [])
    expected = {
        "frequency_domain",
        "structural_patch",
        "dtw_alignment",
        "trend_analysis",
        "quantile_analysis",
        "integrated_signal",
    }
    assert expected.issubset(set(names))
