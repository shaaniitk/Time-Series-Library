from typing import Dict, Any, Type, Optional
from types import SimpleNamespace
import torch.nn as nn
from enum import Enum

class ComponentFamily(Enum):
    """Enumeration of all component families in the modular system."""
    ATTENTION = "attention"
    EMBEDDING = "embedding"
    DECOMPOSITION = "decomposition"
    SAMPLING = "sampling"
    OUTPUT_HEAD = "output_head"
    NORMALIZATION = "normalization"
    BACKBONE = "backbone"
    FEEDFORWARD = "feedforward"
    OUTPUT = "output"
    PROCESSOR = "processor"
    TUNING = "tuning"
    ENCODER = "encoder"
    DECODER = "decoder"
    LOSS = "loss"
    FUSION = "fusion"

class ComponentRegistry:
    """
    A central, singleton-like registry for all neural network components.
    
    This registry holds the component's class, its type, and the default
    configuration needed to instantiate it for testing and default use.
    """
    def __init__(self):
        self._registry: Dict[ComponentFamily, Dict[str, Dict[str, Any]]] = {
            family: {} for family in ComponentFamily
        }

    def register(self, name: str, component_class: type, component_type: ComponentFamily, metadata: dict = None, test_config: dict = None):
        """
        Registers a new component, making it available to the factory and test suite.
        
        Args:
            name (str): The unique name to identify the component.
            component_class (type): The component's class definition.
            component_type (ComponentFamily): The family the component belongs to.
            metadata (dict): Optional component metadata.
            test_config (dict): Keyword arguments needed to instantiate
                                         the class for testing.
        """
        # Backward-compatible argument order shim:
        # Support signature: register(component_type, name, component_class, ...)
        # Some legacy call-sites (deprecation shims) pass (family, name, cls).
        if isinstance(name, ComponentFamily) and isinstance(component_class, str) and isinstance(component_type, type):
            component_type, name, component_class = name, component_class, component_type  # type: ignore

        if name in self._registry[component_type]:
            print(f"Warning: Component '{name}' of type '{component_type.value}' is being overwritten.")
            
        # Prepare metadata with automatic fields
        final_metadata = metadata or {}
        final_metadata.update({
            'class_name': component_class.__name__,
            'module': component_class.__module__,
            'registered_name': name
        })
        
        self._registry[component_type][name] = {
            "class": component_class,
            "config": test_config or {},
            "metadata": final_metadata
        }

    def create(
        self,
        name: str,
        component_type: ComponentFamily,
        **override_kwargs: Any
    ) -> nn.Module:
        """
        Factory method to create an instance of a registered component.
        
        This is the primary way models should instantiate components.
        
        Args:
            name (str): The name of the component to create.
            component_type (ComponentFamily): The family of the component.
            **override_kwargs: Any arguments to override the default test_config.
            
        Returns:
            nn.Module: An instance of the requested component.
        """
        # Backward-compatible argument order shim:
        # Some callers use create(component_type, name, **kwargs). Detect and swap.
        if isinstance(name, ComponentFamily) and isinstance(component_type, str):
            name, component_type = component_type, name  # type: ignore[assignment]
            component_type = ComponentFamily(component_type)  # type: ignore[assignment]

        if name not in self._registry[component_type]:
            raise ValueError(f"Component '{name}' not found in registry for type '{component_type.value}'. "
                             "Ensure it has been registered.")
        
        component_info = self._registry[component_type][name]
        component_class = component_info["class"]
        
        # Start with default config and update with any overrides
        final_config = component_info["config"].copy()
        final_config.update(override_kwargs)
        
        # Filter kwargs to those accepted by the component's constructor
        import inspect
        if isinstance(component_class, type):
            # Class: filter based on __init__
            try:
                sig = inspect.signature(component_class.__init__)
                valid = set(sig.parameters.keys()) - {"self"}
                filtered_config = {k: v for k, v in final_config.items() if k in valid}
            except Exception:
                filtered_config = final_config
            return component_class(**filtered_config)
        else:
            # Callable factory: filter based on callable signature
            try:
                sig = inspect.signature(component_class)
                params = list(sig.parameters.values())
                # If callable accepts **kwargs, pass through all values
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                    filtered_config = final_config
                else:
                    valid = set(sig.parameters.keys())
                    filtered_config = {k: v for k, v in final_config.items() if k in valid}
            except Exception:
                filtered_config = final_config
            return component_class(**filtered_config)

    def get_all_by_type(self, component_type: ComponentFamily) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves all registered components of a specific type.
        Primarily used by the testing framework for discovery.
        """
        return self._registry[component_type].copy()
    
    def list(self, component_type: Optional[ComponentFamily] = None) -> Dict[str, list]:
        """
        Returns a mapping from family name to a list of registered component names.
        
        Backward-compatibility:
        - When a specific ComponentFamily is provided, return a dict with a single key
          (ComponentFamily.value) mapped to the list of names, matching tests that index
          by ['attention'] or [ComponentFamily.ATTENTION.value].
        - When no family is provided, return all families in the mapping.
        """
        if component_type is not None:
            return {component_type.value: list(self._registry[component_type].keys())}
        return {family.value: list(self._registry[family].keys()) for family in ComponentFamily}

    def resolve(self, component_type: ComponentFamily, name: str) -> SimpleNamespace:
        """
        Resolve a registered component to its metadata wrapper.
        Provided for legacy shims that expect `.cls` access.
        """
        if name not in self._registry[component_type]:
            raise ValueError(
                f"Component '{name}' not found in registry for type '{component_type.value}'."
            )
        info = self._registry[component_type][name]
        return SimpleNamespace(cls=info["class"], config=info["config"], metadata=info["metadata"])

# Create a single, global instance of the registry to be imported everywhere.
component_registry = ComponentRegistry()