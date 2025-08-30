import pytest
from layers.modular.core.registry import component_registry, ComponentFamily

def discover_components_from_registry(component_type: ComponentFamily):
    """
    Discovers all components of a given type directly from the registry.
    """
    component_params = []
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