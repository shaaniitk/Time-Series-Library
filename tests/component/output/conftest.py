import pytest
import importlib
from typing import List
from layers.modular.core.registry import component_registry, ComponentFamily

def _load_output_modules() -> List[str]:
    """
    Import output implementation modules so their per-file registrations run.

    We avoid importing the package-level `layers.modular.output` to sidestep
    legacy/stale exports in its __init__. Instead, import known submodules
    directly. Missing modules are ignored gracefully.
    """
    module_names: List[str] = [
        # Core output modules
        "layers.modular.output.linear_output",
        "layers.modular.output.forecasting_head",
        "layers.modular.output.regression_head",
        "layers.modular.output.classification_head",
        "layers.modular.output.multi_task_head",
        "layers.modular.output.attention_pooling_head",
        "layers.modular.output.adaptive_head",
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
    # Ensure output modules are imported so per-file registrations execute
    if component_type == ComponentFamily.OUTPUT:
        _load_output_modules()
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
    # This will find any test that uses the 'output_param' fixture
    if "output_param" in metafunc.fixturenames:
        output_modules = discover_components_from_registry(ComponentFamily.OUTPUT)
        metafunc.parametrize("output_param", output_modules)