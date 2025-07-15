"""
Component Factory System

Provides a centralized factory for creating modular components with proper
type checking and configuration validation.
"""

from typing import Any, Dict, Type, Optional, Union
import logging

from .registry import ComponentRegistry
from .base_interfaces import (
    BaseComponent, BaseBackbone, BaseEmbedding, BaseAttention,
    BaseFeedForward, BaseOutput, BaseLoss, BaseAdapter, BaseProcessor
)

logger = logging.getLogger(__name__)


class ComponentFactory:
    """
    Factory class for creating modular components with proper type checking
    and configuration validation.
    """
    
    def __init__(self):
        self.registry = ComponentRegistry()
        self._type_mapping = {
            'backbone': BaseBackbone,
            'embedding': BaseEmbedding,
            'attention': BaseAttention,
            'feedforward': BaseFeedForward,
            'output': BaseOutput,
            'loss': BaseLoss,
            'adapter': BaseAdapter,
            'processor': BaseProcessor
        }
    
    def create_component(
        self,
        component_type: str,
        component_name: str,
        config: Union[Dict[str, Any], Any],
        **kwargs
    ) -> BaseComponent:
        """
        Create a component instance.
        
        Args:
            component_type: Type of component ('backbone', 'embedding', etc.)
            component_name: Name of the specific implementation
            config: Configuration object or dictionary
            **kwargs: Additional arguments
            
        Returns:
            Instantiated component
            
        Raises:
            ValueError: If component type or name is invalid
            RuntimeError: If component creation fails
        """
        try:
            # Validate component type
            if component_type not in self._type_mapping:
                available_types = list(self._type_mapping.keys())
                raise ValueError(
                    f"Invalid component type '{component_type}'. "
                    f"Available types: {available_types}"
                )
            
            # Get component class from registry
            component_class = self.registry.get_component(component_type, component_name)
            if component_class is None:
                available_components = self.registry.list_components(component_type)
                raise ValueError(
                    f"Component '{component_name}' not found for type '{component_type}'. "
                    f"Available components: {available_components}"
                )
            
            # Validate that class implements the correct interface
            expected_base = self._type_mapping[component_type]
            if not issubclass(component_class, expected_base):
                raise ValueError(
                    f"Component class {component_class} does not implement {expected_base}"
                )
            
            # Handle different config types
            if hasattr(config, '__dict__'):
                # Config object - pass directly
                component = component_class(config, **kwargs)
            elif isinstance(config, dict):
                # Dictionary config - pass directly
                component = component_class(config, **kwargs)
            else:
                # Other types - wrap in dict
                component = component_class({'config': config}, **kwargs)
            
            # Validate the created component
            if not isinstance(component, expected_base):
                raise RuntimeError(
                    f"Created component is not an instance of {expected_base}"
                )
            
            logger.info(
                f"Successfully created {component_type} component: {component_name}"
            )
            
            return component
            
        except Exception as e:
            logger.error(
                f"Failed to create {component_type} component '{component_name}': {e}"
            )
            raise RuntimeError(
                f"Component creation failed: {e}"
            ) from e
    
    def create_backbone(
        self,
        backbone_name: str,
        config: Union[Dict[str, Any], Any],
        **kwargs
    ) -> BaseBackbone:
        """Create a backbone component."""
        return self.create_component('backbone', backbone_name, config, **kwargs)
    
    def create_embedding(
        self,
        embedding_name: str,
        config: Union[Dict[str, Any], Any],
        **kwargs
    ) -> BaseEmbedding:
        """Create an embedding component."""
        return self.create_component('embedding', embedding_name, config, **kwargs)
    
    def create_attention(
        self,
        attention_name: str,
        config: Union[Dict[str, Any], Any],
        **kwargs
    ) -> BaseAttention:
        """Create an attention component."""
        return self.create_component('attention', attention_name, config, **kwargs)
    
    def create_feedforward(
        self,
        feedforward_name: str,
        config: Union[Dict[str, Any], Any],
        **kwargs
    ) -> BaseFeedForward:
        """Create a feedforward component."""
        return self.create_component('feedforward', feedforward_name, config, **kwargs)
    
    def create_output(
        self,
        output_name: str,
        config: Union[Dict[str, Any], Any],
        **kwargs
    ) -> BaseOutput:
        """Create an output component."""
        return self.create_component('output', output_name, config, **kwargs)
    
    def create_loss(
        self,
        loss_name: str,
        config: Union[Dict[str, Any], Any],
        **kwargs
    ) -> BaseLoss:
        """Create a loss component."""
        return self.create_component('loss', loss_name, config, **kwargs)
    
    def create_adapter(
        self,
        adapter_name: str,
        config: Union[Dict[str, Any], Any],
        **kwargs
    ) -> BaseAdapter:
        """Create an adapter component."""
        return self.create_component('adapter', adapter_name, config, **kwargs)
    
    def create_processor(
        self,
        processor_name: str,
        config: Union[Dict[str, Any], Any],
        **kwargs
    ) -> BaseProcessor:
        """Create a processor component."""
        return self.create_component('processor', processor_name, config, **kwargs)
    
    def list_available_components(self) -> Dict[str, list]:
        """
        List all available components by type.
        
        Returns:
            Dictionary mapping component types to lists of available implementations
        """
        available = {}
        for component_type in self._type_mapping.keys():
            available[component_type] = self.registry.list_components(component_type)
        return available
    
    def get_component_info(self, component_type: str, component_name: str) -> Dict[str, Any]:
        """
        Get information about a specific component.
        
        Args:
            component_type: Type of component
            component_name: Name of the implementation
            
        Returns:
            Component information dictionary
        """
        component_class = self.registry.get_component(component_type, component_name)
        if component_class is None:
            return {}
        
        info = {
            'type': component_type,
            'name': component_name,
            'class': component_class.__name__,
            'module': component_class.__module__,
        }
        
        # Try to get additional info from the class
        if hasattr(component_class, 'get_capabilities'):
            try:
                # Create a dummy instance to get capabilities
                dummy_config = {}
                if component_type == 'backbone':
                    dummy_config = {'d_model': 512}
                elif component_type in ['embedding', 'feedforward', 'output']:
                    dummy_config = {'d_model': 512}
                elif component_type == 'attention':
                    dummy_config = {'d_model': 512, 'num_heads': 8}
                elif component_type == 'loss':
                    dummy_config = {'reduction': 'mean'}
                
                if dummy_config:
                    dummy_instance = component_class(dummy_config)
                    info['capabilities'] = dummy_instance.get_capabilities()
                    
                    # Get type-specific info
                    if hasattr(dummy_instance, 'get_backbone_type'):
                        info['backbone_type'] = dummy_instance.get_backbone_type()
                    elif hasattr(dummy_instance, 'get_embedding_type'):
                        info['embedding_type'] = dummy_instance.get_embedding_type()
                    elif hasattr(dummy_instance, 'get_attention_type'):
                        info['attention_type'] = dummy_instance.get_attention_type()
                    elif hasattr(dummy_instance, 'get_ffn_type'):
                        info['ffn_type'] = dummy_instance.get_ffn_type()
                    elif hasattr(dummy_instance, 'get_output_type'):
                        info['output_type'] = dummy_instance.get_output_type()
                    elif hasattr(dummy_instance, 'get_loss_type'):
                        info['loss_type'] = dummy_instance.get_loss_type()
                        
            except Exception as e:
                logger.warning(f"Could not get capabilities for {component_name}: {e}")
        
        return info
    
    def validate_config(
        self,
        component_type: str,
        component_name: str,
        config: Union[Dict[str, Any], Any]
    ) -> bool:
        """
        Validate a configuration for a specific component.
        
        Args:
            component_type: Type of component
            component_name: Name of the implementation
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Try to create the component with the config
            component = self.create_component(component_type, component_name, config)
            return True
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}") from e


# Global factory instance
_factory = None

def get_factory() -> ComponentFactory:
    """Get the global ComponentFactory instance."""
    global _factory
    if _factory is None:
        _factory = ComponentFactory()
    return _factory
