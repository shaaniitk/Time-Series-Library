import pytest
import importlib
from typing import List
from layers.modular.core.registry import component_registry, ComponentFamily

def _load_backbone_modules() -> List[str]:
    """
    Import backbone implementation modules so their per-file registrations run.

    We avoid importing the package-level `layers.modular.backbone` to sidestep
    legacy/stale exports in its __init__. Instead, import known submodules
    directly. Missing modules are ignored gracefully.
    """
    module_names: List[str] = [
        # Core backbone modules
        "layers.modular.backbone.chronos_backbone",
        "layers.modular.backbone.t5_backbone", 
        "layers.modular.backbone.bert_backbone",
        "layers.modular.backbone.simple_transformer",
        "layers.modular.backbone.lstm_backbone",
        "layers.modular.backbone.gru_backbone",
        "layers.modular.backbone.cnn_backbone",
        "layers.modular.backbone.resnet_backbone",
        "layers.modular.backbone.transformer_backbone",
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
    # Ensure backbone modules are imported so per-file registrations execute
    if component_type == ComponentFamily.BACKBONE:
        _load_backbone_modules()
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
    # This will find any test that uses the 'backbone_param' fixture
    if "backbone_param" in metafunc.fixturenames:
        backbone_modules = discover_components_from_registry(ComponentFamily.BACKBONE)
        metafunc.parametrize("backbone_param", backbone_modules)