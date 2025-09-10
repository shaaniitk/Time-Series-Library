"""Output Component Registry

This module provides a registry for output layer components including
forecasting heads, regression heads, and linear projections.
"""

import logging
from typing import Dict, Any, Type, Optional
from dataclasses import dataclass

from ..core.registry import ComponentRegistry, ComponentFamily
from ..base_interfaces import BaseOutput
from .linear_output import LinearOutput, LinearOutputConfig
from .outputs import ForecastingHead, RegressionHead, OutputConfig

logger = logging.getLogger(__name__)


class OutputRegistry(ComponentRegistry):
    """Registry for output layer components"""
    
    def __init__(self):
        super().__init__(ComponentFamily.OUTPUT)
        self._register_components()
    
    def _register_components(self):
        """Register all available output components"""
        
        # Linear output components
        self.register_component(
            name="linear",
            component_class=LinearOutput,
            config_class=LinearOutputConfig,
            description="Simple linear projection output layer",
            tags=["linear", "projection", "basic"]
        )
        
        # Forecasting heads
        self.register_component(
            name="forecasting",
            component_class=ForecastingHead,
            config_class=OutputConfig,
            description="Output head for time series forecasting tasks",
            tags=["forecasting", "timeseries", "prediction"]
        )
        
        # Regression heads
        self.register_component(
            name="regression",
            component_class=RegressionHead,
            config_class=OutputConfig,
            description="Output head for regression tasks",
            tags=["regression", "continuous", "prediction"]
        )
        
        logger.info(f"Registered {len(self._components)} output components")
    
    def create_component(self, name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> BaseOutput:
        """Create an output component instance
        
        Args:
            name: Component name
            config: Configuration dictionary
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured output component instance
        """
        if name not in self._components:
            available = list(self._components.keys())
            raise ValueError(f"Unknown output component '{name}'. Available: {available}")
        
        component_info = self._components[name]
        config_class = component_info['config_class']
        component_class = component_info['component_class']
        
        # Merge config dict with kwargs
        if config is None:
            config = {}
        config.update(kwargs)
        
        # Create config instance
        if config_class == LinearOutputConfig:
            # Handle LinearOutputConfig parameters
            config_instance = config_class(
                d_model=config.get('d_model', 512),
                output_dim=config.get('output_dim', 1)
            )
        elif config_class == OutputConfig:
            # Handle OutputConfig parameters
            config_instance = config_class(
                d_model=config.get('d_model', 512),
                output_dim=config.get('output_dim', 1),
                horizon=config.get('horizon', 1),
                dropout=config.get('dropout', 0.1),
                use_bias=config.get('use_bias', True),
                activation=config.get('activation', None)
            )
        else:
            # Generic config creation
            config_instance = config_class(**config)
        
        # Create component instance
        component = component_class(config_instance)
        
        logger.info(f"Created output component '{name}' with config: {config}")
        return component
    
    def get_component_types(self) -> Dict[str, str]:
        """Get mapping of component names to their types"""
        return {
            name: info['description']
            for name, info in self._components.items()
        }
    
    def get_components_by_tag(self, tag: str) -> Dict[str, Dict[str, Any]]:
        """Get components that have a specific tag"""
        return {
            name: info
            for name, info in self._components.items()
            if tag in info.get('tags', [])
        }


# Global registry instance
_output_registry = OutputRegistry()


def get_output_component(name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> BaseOutput:
    """Factory function to create output components
    
    Args:
        name: Component name ('linear', 'forecasting', 'regression')
        config: Configuration dictionary
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured output component instance
        
    Example:
        >>> # Create a linear output
        >>> linear_out = get_output_component('linear', d_model=256, output_dim=1)
        >>> 
        >>> # Create a forecasting head
        >>> forecast_head = get_output_component('forecasting', 
        ...                                     d_model=512, 
        ...                                     output_dim=1, 
        ...                                     horizon=24)
        >>> 
        >>> # Create a regression head
        >>> regression_head = get_output_component('regression',
        ...                                       d_model=512,
        ...                                       output_dim=3,
        ...                                       dropout=0.2)
    """
    return _output_registry.create_component(name, config, **kwargs)


def list_output_components() -> Dict[str, str]:
    """List all available output components"""
    return _output_registry.get_component_types()


def get_output_components_by_tag(tag: str) -> Dict[str, Dict[str, Any]]:
    """Get output components by tag"""
    return _output_registry.get_components_by_tag(tag)


# Register with unified registry
try:
    from ..core.registry import unified_registry
    unified_registry.register_family_registry(ComponentFamily.OUTPUT, _output_registry)
    logger.info("Output registry registered with unified registry")
except ImportError:
    logger.warning("Could not register with unified registry")


__all__ = [
    'OutputRegistry',
    'get_output_component', 
    'list_output_components',
    'get_output_components_by_tag'
]