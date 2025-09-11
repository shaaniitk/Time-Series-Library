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
        super().__init__()
        self._register_components()
    
    def _register_components(self):
        """Register all available output components"""
        
        # Linear output components
        self.register(
            "linear",
            LinearOutput,
            ComponentFamily.OUTPUT,
            {
                "config_class": LinearOutputConfig,
                "description": "Simple linear projection output layer",
                "tags": ["linear", "projection", "basic"]
            }
        )
        
        # Forecasting heads
        self.register(
            "forecasting",
            ForecastingHead,
            ComponentFamily.OUTPUT,
            {
                "config_class": OutputConfig,
                "description": "Output head for time series forecasting tasks",
                "tags": ["forecasting", "timeseries", "prediction"]
            }
        )
        
        # Regression heads
        self.register(
            "regression",
            RegressionHead,
            ComponentFamily.OUTPUT,
            {
                "config_class": OutputConfig,
                "description": "Output head for regression tasks",
                "tags": ["regression", "continuous", "prediction"]
            }
        )
        
        logger.info(f"Registered output components successfully")
    
    def create_component(self, name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> BaseOutput:
        """Create an output component by name."""
        # Merge config with kwargs
        final_config = {}
        if config:
            final_config.update(config)
        final_config.update(kwargs)
        
        return self.create(name, ComponentFamily.OUTPUT, **final_config)
    
    def get_component_types(self) -> Dict[str, str]:
        """Get all available output component types."""
        components = self.get_all_by_type(ComponentFamily.OUTPUT)
        return {name: info.get('description', 'No description') for name, info in components.items()}
    
    def get_components_by_tag(self, tag: str) -> Dict[str, Dict[str, Any]]:
        """Get all components that have a specific tag."""
        components = self.get_all_by_type(ComponentFamily.OUTPUT)
        return {
            name: info for name, info in components.items()
            if tag in info.get('config', {}).get('tags', [])
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