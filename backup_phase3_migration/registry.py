"""
Component Registry for Dynamic Component Management

This registry allows for dynamic registration and discovery of components,
enabling easy extension and plugin-like architecture.
"""

import logging
import torch.nn as nn
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
    
    def get_component_info(self, component_type: str, component_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a component
        
        Args:
            component_type: Type of component
            component_name: Name of the component
            
        Returns:
            Dictionary with component information including metadata and capabilities
        """
        if not self.is_registered(component_type, component_name):
            return {'error': f'Component {component_type}:{component_name} not found'}
        
        # Get component class and metadata
        component_class = self.get(component_type, component_name)
        metadata = self.get_metadata(component_type, component_name)
        
        # Extract capabilities and requirements if available
        info = {
            'component_type': component_type,
            'component_name': component_name,
            'class_name': component_class.__name__,
            'metadata': metadata,
        }
        
        # Add capabilities and requirements if component supports them
        if hasattr(component_class, 'get_capabilities'):
            try:
                info['capabilities'] = component_class.get_capabilities()
            except:
                info['capabilities'] = []
        
        if hasattr(component_class, 'get_requirements'):
            try:
                info['requirements'] = component_class.get_requirements()
            except:
                info['requirements'] = []
        
        return info
    
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


def create_global_registry() -> ComponentRegistry:
    """
    Create and initialize the global registry with components
    
    Returns:
        Initialized ComponentRegistry
    """
    # Clear existing registry
    _global_registry.clear()
    
    # Register example components
    try:
        from .example_components import register_example_components
        register_example_components(_global_registry)
        logger.info("Global registry initialized with example components")
    except ImportError as e:
        logger.warning(f"Could not load example components: {e}")
    
    # Try to register components from existing modules
    try:
        _register_fallback_components(_global_registry)
        logger.info("✅ Registered fallback components")
    except Exception as e:
        logger.warning(f"Could not register fallback components: {e}")
    
    return _global_registry


def _register_fallback_components(registry: ComponentRegistry):
    """Register basic fallback components"""
    
    # Simple fallback components that can be used for testing
    class MockBackbone(BaseComponent):
        def __init__(self, config):
            super().__init__(config)
            self.d_model = getattr(config, 'd_model', 512)
        
        def forward(self, x, **kwargs):
            return x
        
        def get_output_dim(self):
            return self.d_model
        
        @classmethod
        def get_capabilities(cls):
            return ['mock', 'testing']
    
    class MockProcessor(BaseComponent):
        def __init__(self, config):
            super().__init__(config)
            self.d_model = getattr(config, 'd_model', 512)
        
        def forward(self, embedded_input, backbone_output, target_length, **kwargs):
            # Simple repeat to target length
            if backbone_output.size(1) != target_length:
                pooled = backbone_output.mean(dim=1, keepdim=True)
                return pooled.repeat(1, target_length, 1)
            return backbone_output
        
        def get_output_dim(self):
            return self.d_model
        
        @classmethod
        def get_capabilities(cls):
            return ['mock', 'testing']
    
    class MockLoss(BaseComponent):
        def __init__(self, config):
            super().__init__(config)
            self.loss_fn = nn.MSELoss()
        
        def forward(self, predictions, targets, **kwargs):
            return self.loss_fn(predictions, targets)
        
        def get_output_dim(self):
            return 1
        
        @classmethod
        def get_capabilities(cls):
            return ['mock', 'testing', 'mse']
    
    # Register fallback components
    registry.register('backbone', 'mock_backbone', MockBackbone)
    registry.register('processor', 'mock_processor', MockProcessor) 
    registry.register('loss', 'mock_loss', MockLoss)
