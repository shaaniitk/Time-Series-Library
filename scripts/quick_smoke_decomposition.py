import torch

from unified_component_registry import unified_registry
from utils.implementations.decomposition.wrapped_decompositions import DecompositionProcessorConfig


def run_smoke():
    print("Listing processors in unified registry...")
    comps = unified_registry.list_all_components()
    print({k: len(v) for k, v in comps.items()})

    x = torch.randn(2, 64, 32)

    # Series decomposition processor
    cfg = DecompositionProcessorConfig(d_model=32, kernel_size=7, component_selection="seasonal")
    proc = unified_registry.create_component('processor', 'series_decomposition_processor', cfg)
    y = proc.process_sequence(x, backbone_output=None, target_length=64)
    assert y.shape == x.shape
    print("SeriesDecompositionProcessor OK", y.shape)

    # Learnable decomposition processor
    lcfg = DecompositionProcessorConfig(d_model=32, kernel_size=15, component_selection="trend")
    lproc = unified_registry.create_component('processor', 'learnable_series_decomposition_processor', lcfg)
    y2 = lproc.process_sequence(x, backbone_output=None, target_length=64)
    assert y2.shape == x.shape
    print("LearnableDecompositionProcessor OK", y2.shape)

    # Wavelet hierarchical decomposition processor
    wcfg = DecompositionProcessorConfig(d_model=32, component_selection="concat", seq_len=64, levels=2)
    wproc = unified_registry.create_component('processor', 'wavelet_hierarchical_decomposition_processor', wcfg)
    y3 = wproc.process_sequence(x, backbone_output=None, target_length=64)
    assert y3.shape == x.shape
    print("WaveletDecompositionProcessor OK", y3.shape)

    print("\nDECOMPOSITION SMOKE CHECK: SUCCESS")


if __name__ == "__main__":
    run_smoke()
