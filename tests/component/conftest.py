import pytest
import importlib
from typing import List
from layers.modular.core.registry import component_registry, ComponentFamily

def _load_attention_modules() -> List[str]:
    """
    Import attention implementation modules so their per-file registrations run.

    We avoid importing the package-level `layers.modular.attention` to sidestep
    legacy/stale exports in its __init__. Instead, import known submodules
    directly. Missing modules are ignored gracefully.
    """
    module_names: List[str] = [
        # Core
        "layers.modular.attention.core.multihead_attention",
        # Fourier / Frequency
        "layers.modular.attention.fourier.fourier_attention",
        "layers.modular.attention.frequency.fourier_cross_attention",
        "layers.modular.attention.frequency.fourier_block",
        # Wavelet
        "layers.modular.attention.wavelet.wavelet_attention",
        "layers.modular.attention.wavelet.multiscale_wavelet_attention",
        # Sparse
        "layers.modular.attention.sparse.sparse_attention",
        "layers.modular.attention.sparse.logsparse_attention",
        "layers.modular.attention.sparse.probsparse_attention",
        # Time series autocorrelation
        "layers.modular.attention.timeseries.autocorrelation",
        "layers.modular.attention.timeseries.hierarchchical_autocorrelation",
        # Temporal conv variants
        "layers.modular.attention.temporal_conv.causal_convolution",
        "layers.modular.attention.temporal_conv.convolutional_attention",
        "layers.modular.attention.temporal_conv.temporal_conv_net",
        # Bayesian
        "layers.modular.attention.bayesian.bayesian_attention",
        "layers.modular.attention.bayesian.bayesian_multi_head_attention",
        "layers.modular.attention.bayesian.bayesian_cross_attention",
        # Adaptive
        "layers.modular.attention.adaptive.adaptive_mixture",
        # Specialized
        "layers.modular.attention.specialized.cross_resolution_attention",
        # Graph (file may be non-standard name/extension; best-effort)
        "layers.modular.attention.graph.graph_attention_layer",
    ]

    imported: List[str] = []
    for mod in module_names:
        try:
            importlib.import_module(mod)
            imported.append(mod)
        except Exception:
            # Best-effort: ignore modules that fail to import in current env
            continue
    return imported

def discover_components_from_registry(component_type: ComponentFamily):
    """
    Discovers all components of a given type directly from the registry.
    """
    component_params = []
    # Ensure attention modules are imported so per-file registrations execute
    if component_type == ComponentFamily.ATTENTION:
        _load_attention_modules()
    registered_components = component_registry.get_all_by_type(component_type)
    
    for name, info in registered_components.items():
        param = pytest.param(
            (info["class"], info["config"]),
            id=name  # Use the registration name for clear test reports
        )
        component_params.append(param)
            
    return component_params

# Pytest hook that parametrizes test functions
def pytest_generate_tests(metafunc):
    # This will find any test that uses the 'attention_param' fixture
    if "attention_param" in metafunc.fixturenames:
        attention_modules = discover_components_from_registry(ComponentFamily.ATTENTION)
        metafunc.parametrize("attention_param", attention_modules)