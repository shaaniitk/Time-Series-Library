import pytest
import importlib
from typing import List
from layers.modular.core.registry import component_registry, ComponentFamily

def _load_feedforward_modules() -> List[str]:
    """
    Import feedforward implementation modules so their per-file registrations run.

    We avoid importing the package-level `layers.modular.feedforward` to sidestep
    legacy/stale exports in its __init__. Instead, import known submodules
    directly. Missing modules are ignored gracefully.
    """
    module_names: List[str] = [
        # Core feedforward modules
        "layers.modular.feedforward.standard_ffn",
        "layers.modular.feedforward.gated_ffn",
        "layers.modular.feedforward.moe_ffn",
        "layers.modular.feedforward.conv_ffn",
        "layers.modular.feedforward.residual_ffn",
        "layers.modular.feedforward.highway_ffn",
        "layers.modular.feedforward.swish_ffn",
        "layers.modular.feedforward.gelu_ffn",
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
    # Ensure feedforward modules are imported so per-file registrations execute
    if component_type == ComponentFamily.FEEDFORWARD:
        _load_feedforward_modules()
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
    # This will find any test that uses the 'feedforward_param' fixture
    if "feedforward_param" in metafunc.fixturenames:
        feedforward_modules = discover_components_from_registry(ComponentFamily.FEEDFORWARD)
        metafunc.parametrize("feedforward_param", feedforward_modules)