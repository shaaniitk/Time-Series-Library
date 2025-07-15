"""
Component Registry for Dynamic Component Management

This registry allows for dynamic registration and discovery of components,
enabling easy extension and plugin-like architecture.
"""

import logging
from typing import Dict, Type, Any, List, Optional
from .base_interfaces import ComponentType, BaseComponent
from .config_schemas import ComponentConfig

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Central registry for managing modular components
    
    This registry allows for dynamic registration, discovery, and instantiation
    of components, enabling a plugin-like architecture for model building.
    """
    
    def __init__(self):
        self._components: Dict[str, Dict[str, Type[BaseComponent]]] = {
            'backbone': {},
            'embedding': {},
            'attention': {},
            'processor': {},
            'feedforward': {},
            'loss': {},
            'output': {},
            'adapter': {}
        }
        self._metadata: Dict[str, Dict[str, Dict[str, Any]]] = {
            'backbone': {},
            'embedding': {},
            'attention': {},
            'processor': {},
            'feedforward': {},
            'loss': {},
            'output': {},
            'adapter': {}
        }
    
    def register(self, 
                component_type: str, 
                component_name: str, 
                component_class: Type[BaseComponent],
                metadata: Optional[Dict[str, Any]] = None):
        """
        Register a component in the registry
        
        Args:
            component_type: Type of component ('backbone', 'embedding', etc.)
            component_name: Unique name for the component
            component_class: Class implementing the component
            metadata: Optional metadata about the component
        """
        if component_type not in self._components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        if component_name in self._components[component_type]:
            logger.warning(f"Overwriting existing {component_type} component: {component_name}")
        
        self._components[component_type][component_name] = component_class
        
        # Store metadata
        component_metadata = metadata or {}
        component_metadata.update({
            'class_name': component_class.__name__,
            'module': component_class.__module__,
            'registered_name': component_name
        })
        self._metadata[component_type][component_name] = component_metadata
        
        logger.info(f"Registered {component_type} component: {component_name}")
    
    def get(self, component_type: str, component_name: str) -> Type[BaseComponent]:
        """
        Get a component class from the registry
        
        Args:
            component_type: Type of component
            component_name: Name of the component
            
        Returns:
            Component class
        """
        if component_type not in self._components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        if component_name not in self._components[component_type]:
            available = list(self._components[component_type].keys())
            raise ValueError(f"Component '{component_name}' not found in {component_type}. "
                           f"Available: {available}")
        
        return self._components[component_type][component_name]
    
    def create(self, 
              component_type: str, 
              component_name: str, 
              config: ComponentConfig) -> BaseComponent:
        """
        Create an instance of a component
        
        Args:
            component_type: Type of component
            component_name: Name of the component
            config: Configuration for the component
            
        Returns:
            Instantiated component
        """
        component_class = self.get(component_type, component_name)
        
        try:
            return component_class(config)
        except Exception as e:
            logger.error(f"Failed to create {component_type} component '{component_name}': {e}")
            raise
    
    def list_components(self, component_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all registered components
        
        Args:
            component_type: Optional component type to filter by
            
        Returns:
            Dictionary mapping component types to lists of component names
        """
        if component_type:
            if component_type not in self._components:
                raise ValueError(f"Unknown component type: {component_type}")
            return {component_type: list(self._components[component_type].keys())}
        
        return {
            comp_type: list(components.keys())
            for comp_type, components in self._components.items()
        }
    
    def get_metadata(self, component_type: str, component_name: str) -> Dict[str, Any]:
        """
        Get metadata for a component
        
        Args:
            component_type: Type of component
            component_name: Name of the component
            
        Returns:
            Component metadata
        """
        if component_type not in self._metadata:
            raise ValueError(f"Unknown component type: {component_type}")
        
        if component_name not in self._metadata[component_type]:
            raise ValueError(f"Component '{component_name}' not found in {component_type}")
        
        return self._metadata[component_type][component_name].copy()
    
    def unregister(self, component_type: str, component_name: str):
        """
        Unregister a component
        
        Args:
            component_type: Type of component
            component_name: Name of the component
        """
        if component_type not in self._components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        if component_name not in self._components[component_type]:
            logger.warning(f"Component '{component_name}' not found in {component_type}")
            return
        
        del self._components[component_type][component_name]
        del self._metadata[component_type][component_name]
        
        logger.info(f"Unregistered {component_type} component: {component_name}")
    
    def is_registered(self, component_type: str, component_name: str) -> bool:
        """
        Check if a component is registered
        
        Args:
            component_type: Type of component
            component_name: Name of the component
            
        Returns:
            True if component is registered, False otherwise
        """
        return (component_type in self._components and 
                component_name in self._components[component_type])
    
    def get_compatible_components(self, 
                                 component_type: str, 
                                 requirements: Dict[str, Any]) -> List[str]:
        """
        Get components that meet certain requirements
        
        Args:
            component_type: Type of component
            requirements: Dictionary of requirements to match
            
        Returns:
            List of compatible component names
        """
        if component_type not in self._metadata:
            return []
        
        compatible = []
        for name, metadata in self._metadata[component_type].items():
            meets_requirements = True
            for req_key, req_value in requirements.items():
                if req_key not in metadata or metadata[req_key] != req_value:
                    meets_requirements = False
                    break
            
            if meets_requirements:
                compatible.append(name)
        
        return compatible
    
    def clear(self, component_type: Optional[str] = None):
        """
        Clear components from registry
        
        Args:
            component_type: Optional component type to clear (clears all if None)
        """
        if component_type:
            if component_type in self._components:
                self._components[component_type].clear()
                self._metadata[component_type].clear()
                logger.info(f"Cleared all {component_type} components")
        else:
            for comp_type in self._components:
                self._components[comp_type].clear()
                self._metadata[comp_type].clear()
            logger.info("Cleared all components from registry")


# Global registry instance
_global_registry = ComponentRegistry()


def register_component(component_type: str, 
                      component_name: str, 
                      component_class: Type[BaseComponent],
                      metadata: Optional[Dict[str, Any]] = None):
    """
    Convenience function to register a component in the global registry
    
    Args:
        component_type: Type of component
        component_name: Unique name for the component
        component_class: Class implementing the component
        metadata: Optional metadata about the component
    """
    _global_registry.register(component_type, component_name, component_class, metadata)


def get_component(component_type: str, component_name: str) -> Type[BaseComponent]:
    """
    Convenience function to get a component from the global registry
    
    Args:
        component_type: Type of component
        component_name: Name of the component
        
    Returns:
        Component class
    """
    return _global_registry.get(component_type, component_name)


def create_component(component_type: str, 
                    component_name: str, 
                    config: ComponentConfig) -> BaseComponent:
    """
    Convenience function to create a component from the global registry
    
    Args:
        component_type: Type of component
        component_name: Name of the component
        config: Configuration for the component
        
    Returns:
        Instantiated component
    """
    return _global_registry.create(component_type, component_name, config)


def list_all_components() -> Dict[str, List[str]]:
    """
    Convenience function to list all components in the global registry
    
    Returns:
        Dictionary mapping component types to lists of component names
    """
    return _global_registry.list_components()


# Make global registry accessible
def get_global_registry() -> ComponentRegistry:
    """Get the global component registry"""
    return _global_registry
