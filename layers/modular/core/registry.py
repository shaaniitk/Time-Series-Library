from typing import Dict, Any, Type
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
        if name not in self._registry[component_type]:
            raise ValueError(f"Component '{name}' not found in registry for type '{component_type.value}'. "
                             "Ensure it has been registered.")
        
        component_info = self._registry[component_type][name]
        component_class = component_info["class"]
        
        # Start with default config and update with any overrides
        final_config = component_info["config"].copy()
        final_config.update(override_kwargs)

        # Pass config as a single dictionary parameter
        return component_class(config=final_config)

    def get_all_by_type(self, component_type: ComponentFamily) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves all registered components of a specific type.
        Primarily used by the testing framework for discovery.
        """
        return self._registry[component_type].copy()
    
    def list(self, component_type: ComponentFamily) -> list:
        """
        Returns a list of all registered component names for a specific type.
        Used by tests to enumerate available components.
        """
        return list(self._registry[component_type].keys())

# Create a single, global instance of the registry to be imported everywhere.
component_registry = ComponentRegistry()