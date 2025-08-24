"""
Quick smoke test script for FeedForward networks and Adapters via unified registry.
"""
from __future__ import annotations

import torch
from unified_component_registry import unified_registry
from layers.modular.core.config_schemas import FFNConfig


def smoke_ffn() -> None:
    comps = unified_registry.list_all_components()
    print("feedforward:", comps.get("feedforward"))

    cfg = FFNConfig(d_model=32, d_ff=64, dropout=0.1)

    std_cls = unified_registry.get_component("feedforward", "standard_ffn")
    gated_cls = unified_registry.get_component("feedforward", "gated_ffn")
    moe_cls = unified_registry.get_component("feedforward", "moe_ffn")
    conv_cls = unified_registry.get_component("feedforward", "conv_ffn")

    std = std_cls(cfg)
    gated = gated_cls(cfg)
    moe = moe_cls(cfg, num_experts=4, num_selected=2)
    conv = conv_cls(cfg, kernel_size=3)

    x = torch.randn(2, 5, 32)
    for name, mod in [("std", std), ("gated", gated), ("moe", moe), ("conv", conv)]:
        y = mod.apply_feedforward(x)
        print(name, y.shape)
        assert y.shape == x.shape


def smoke_adapters() -> None:
    comps = unified_registry.list_all_components()
    print("adapter:", comps.get("adapter"))

    # Use simple_transformer backbone for adapter smoke
    from layers.modular.core.registry import unified_registry as get_global_registry

    reg = get_global_registry()
    bb_cls = reg.get("backbone", "simple_transformer")

    class BBConfig:
        d_model = 16
        dropout = 0.1
        input_dim = 1

    bb = bb_cls(BBConfig())

    # CovariateAdapter
    from layers.modular.backbone.adapters import CovariateAdapter, MultiScaleAdapter

    cov = CovariateAdapter(bb, {"covariate_dim": 3, "fusion_method": "project", "embedding_dim": 16})
    x_ts = torch.randn(2, 5, 1)
    x_cov = torch.randn(2, 5, 3)
    out = cov(x_ts, x_covariates=x_cov)
    print("covariate_adapter output shape:", out.shape if isinstance(out, torch.Tensor) else type(out))

        # MultiScaleAdapter
    msa = MultiScaleAdapter(bb, {"scales": [1, 2], "aggregation": "concat"})
    # Provide input_dim=1 expected by backbone projection
    x_embed = torch.randn(2, 5, 1)
    out2 = msa(x_embed)
    print("multiscale_adapter output shape:", out2.shape)
    # MultiScale adapter may change channel dim due to concat->linear aggregation
    assert out2.shape[0] == x_embed.shape[0] and out2.shape[1] == x_embed.shape[1]


if __name__ == "__main__":
    smoke_ffn()
    smoke_adapters()
    print("OK")
