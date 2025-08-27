"""
Quick smoke test for modular fusion processors registered via unified registry.
"""
import torch
from tools.unified_component_registry import unified_registry
from layers.modular.processor.wrapped_fusions import FusionProcessorConfig

def main() -> None:
    d_model = 64
    n_levels = 3
    seq_len = 32
    batch = 2
    # Create multi-resolution features
    multi_res_features = [torch.randn(batch, seq_len // (2 ** i), d_model) for i in range(n_levels)]
    base_cfg = FusionProcessorConfig(d_model=d_model, n_levels=n_levels, fusion_strategy='weighted_concat')
    fusion_cls = unified_registry.get_component('processor', 'fusion_hierarchical_processor')
    fusion = fusion_cls(base_cfg)
    out = fusion.process_sequence(multi_res_features, None, target_length=seq_len)
    assert out.shape == (batch, seq_len, d_model), f"fusion_hierarchical_processor produced {out.shape}"
    print(f"fusion_hierarchical_processor: OK with shape {out.shape}")
    print("FUSION SMOKE CHECK: SUCCESS")

if __name__ == "__main__":
    main()
