"""
FeedForward registrations that expose utils FFN implementations to the unified registry.
"""
from __future__ import annotations

from layers.modular.core.registry import register_component
from layers.modular.feedforward import (
    FFNConfig,
    StandardFFN,
    GatedFFN,
    MoEFFN,
    ConvFFN,
)


def register_utils_feedforwards() -> None:
    """Register FeedForward components into the global ComponentRegistry."""
    register_component(
        "feedforward",
        "standard_ffn",
        StandardFFN,
        metadata={
            "domain": "feedforward",
            "features": ["linear", "dropout", "activation"],
            "source": "layers.modular.feedforward",
        },
    )
    register_component(
        "feedforward",
        "gated_ffn",
        GatedFFN,
        metadata={
            "domain": "feedforward",
            "features": ["gated", "glu", "dropout"],
            "source": "layers.modular.feedforward",
        },
    )
    register_component(
        "feedforward",
        "moe_ffn",
        MoEFFN,
        metadata={
            "domain": "feedforward",
            "features": ["moe", "experts", "topk_gating"],
            "source": "layers.modular.feedforward",
        },
    )
    register_component(
        "feedforward",
        "conv_ffn",
        ConvFFN,
        metadata={
            "domain": "feedforward",
            "features": ["conv1d", "same_padding", "dropout"],
            "source": "layers.modular.feedforward",
        },
    )
