"""
Expert Registry System

Centralized registry for managing and discovering different types of experts
in the Mixture of Experts framework.
"""

import torch
import torch.nn as nn
from typing import Dict, Type, Any, List, Optional, Callable
from abc import ABC, abstractmethod

from .base_expert import BaseExpert, ExpertOutput

# Import all expert types
from .temporal.seasonal_expert import SeasonalPatternExpert
from .temporal.trend_expert import TrendPatternExpert
from .temporal.volatility_expert import VolatilityPatternExpert
from .temporal.regime_expert import RegimePatternExpert

from .spatial.local_expert import LocalSpatialExpert
from .spatial.global_expert import GlobalSpatialExpert
from .spatial.hierarchical_expert import HierarchicalSpatialExpert

from .uncertainty.aleatoric_expert import AleatoricUncertaintyExpert
from .uncertainty.epistemic_expert import EpistemicUncertaintyExpert


class ExpertRegistry:
    """
    Registry for managing expert types and their instantiation.
    
    Provides a centralized way to register, discover, and create experts
    for different aspects of time series modeling.
    """
    
    def __init__(self):
        self._experts: Dict[str, Type[BaseExpert]] = {}
        self._expert_categories: Dict[str, List[str]] = {
            'temporal': [],
            'spatial': [],
            'uncertainty': [],
            'multimodal': [],
            'adaptive': []
        }
        
        # Register default experts
        self._register_default_experts()
    
    def _register_default_experts(self):
        """Register all default expert types."""
        
        # Temporal experts
        self.register('seasonal_expert', SeasonalPatternExpert, 'temporal')
        self.register('trend_expert', TrendPatternExpert, 'temporal')
        self.register('volatility_expert', VolatilityPatternExpert, 'temporal')
        self.register('regime_expert', RegimePatternExpert, 'temporal')
        
        # Spatial experts
        self.register('local_spatial_expert', LocalSpatialExpert, 'spatial')
        self.register('global_spatial_expert', GlobalSpatialExpert, 'spatial')
        self.register('hierarchical_spatial_expert', HierarchicalSpatialExpert, 'spatial')
        
        # Uncertainty experts
        self.register('aleatoric_uncertainty_expert', AleatoricUncertaintyExpert, 'uncertainty')
        self.register('epistemic_uncertainty_expert', EpistemicUncertaintyExpert, 'uncertainty')
    
    def register(self, name: str, expert_class: Type[BaseExpert], category: str = 'multimodal'):
        """
        Register a new expert type.
        
        Args:
            name: Unique name for the expert
            expert_class: Expert class (must inherit from BaseExpert)
            category: Expert category ('temporal', 'spatial', 'uncertainty', etc.)
        """
        if not issubclass(expert_class, BaseExpert):
            raise ValueError(f"Expert class {expert_class} must inherit from BaseExpert")
        
        if name in self._experts:
            raise ValueError(f"Expert '{name}' is already registered")
        
        self._experts[name] = expert_class
        
        if category not in self._expert_categories:
            self._expert_categories[category] = []
        
        self._expert_categories[category].append(name)
    
    def get(self, name: str) -> Type[BaseExpert]:
        """Get expert class by name."""
        if name not in self._experts:
            raise ValueError(f"Expert '{name}' not found in registry. Available: {list(self._experts.keys())}")
        
        return self._experts[name]
    
    def create(self, name: str, config: Any, **kwargs) -> BaseExpert:
        """Create an expert instance."""
        expert_class = self.get(name)
        return expert_class(config, **kwargs)
    
    def list_experts(self, category: Optional[str] = None) -> List[str]:
        """List available experts, optionally filtered by category."""
        if category is None:
            return list(self._experts.keys())
        
        if category not in self._expert_categories:
            raise ValueError(f"Category '{category}' not found. Available: {list(self._expert_categories.keys())}")
        
        return self._expert_categories[category].copy()
    
    def list_categories(self) -> List[str]:
        """List available expert categories."""
        return list(self._expert_categories.keys())
    
    def get_expert_info(self, name: str) -> Dict[str, Any]:
        """Get information about a specific expert."""
        expert_class = self.get(name)
        
        # Find category
        category = 'unknown'
        for cat, experts in self._expert_categories.items():
            if name in experts:
                category = cat
                break
        
        return {
            'name': name,
            'class': expert_class.__name__,
            'module': expert_class.__module__,
            'category': category,
            'docstring': expert_class.__doc__,
            'base_classes': [cls.__name__ for cls in expert_class.__mro__[1:]]
        }
    
    def create_expert_ensemble(self, expert_names: List[str], config: Any, **kwargs) -> List[BaseExpert]:
        """Create an ensemble of experts."""
        experts = []
        for name in expert_names:
            expert = self.create(name, config, **kwargs)
            experts.append(expert)
        return experts
    
    def create_category_ensemble(self, category: str, config: Any, **kwargs) -> List[BaseExpert]:
        """Create an ensemble of all experts in a category."""
        expert_names = self.list_experts(category)
        return self.create_expert_ensemble(expert_names, config, **kwargs)
    
    def suggest_experts(self, task_type: str, data_characteristics: Dict[str, Any]) -> List[str]:
        """
        Suggest appropriate experts based on task type and data characteristics.
        
        Args:
            task_type: Type of task ('forecasting', 'classification', 'anomaly_detection')
            data_characteristics: Dictionary with data properties
            
        Returns:
            List of suggested expert names
        """
        suggestions = []
        
        # Task-based suggestions
        if task_type in ['forecasting', 'prediction']:
            suggestions.extend(['trend_expert', 'seasonal_expert'])
            
            if data_characteristics.get('high_volatility', False):
                suggestions.append('volatility_expert')
            
            if data_characteristics.get('regime_changes', False):
                suggestions.append('regime_expert')
        
        # Data characteristic-based suggestions
        if data_characteristics.get('spatial_dependencies', False):
            if data_characteristics.get('local_patterns', True):
                suggestions.append('local_spatial_expert')
            
            if data_characteristics.get('global_patterns', False):
                suggestions.append('global_spatial_expert')
            
            if data_characteristics.get('hierarchical_structure', False):
                suggestions.append('hierarchical_spatial_expert')
        
        # Uncertainty requirements
        if data_characteristics.get('uncertainty_quantification', False):
            suggestions.extend(['aleatoric_uncertainty_expert', 'epistemic_uncertainty_expert'])
        
        return list(set(suggestions))  # Remove duplicates
    
    def validate_expert_compatibility(self, expert_names: List[str], config: Any) -> Dict[str, Any]:
        """
        Validate that experts are compatible with each other and the configuration.
        
        Args:
            expert_names: List of expert names to validate
            config: Model configuration
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'compatible': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        # Check for conflicting experts
        categories = {}
        for name in expert_names:
            expert_info = self.get_expert_info(name)
            category = expert_info['category']
            
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        
        # Check for potential conflicts
        if len(categories.get('temporal', [])) > 2:
            validation_results['warnings'].append(
                f"Multiple temporal experts selected: {categories['temporal']}. "
                "This may lead to redundancy."
            )
        
        if len(categories.get('uncertainty', [])) > 1:
            validation_results['suggestions'].append(
                "Multiple uncertainty experts can provide complementary uncertainty estimates."
            )
        
        # Check configuration compatibility
        required_params = {
            'seasonal_expert': ['seq_len'],
            'trend_expert': ['seq_len', 'd_model'],
            'regime_expert': ['d_model'],
            'volatility_expert': ['d_model']
        }
        
        for name in expert_names:
            if name in required_params:
                for param in required_params[name]:
                    if not hasattr(config, param):
                        validation_results['errors'].append(
                            f"Expert '{name}' requires config parameter '{param}'"
                        )
                        validation_results['compatible'] = False
        
        return validation_results


# Global registry instance
expert_registry = ExpertRegistry()


def register_expert(name: str, category: str = 'multimodal'):
    """
    Decorator for registering expert classes.
    
    Usage:
        @register_expert('my_custom_expert', 'temporal')
        class MyCustomExpert(TemporalExpert):
            ...
    """
    def decorator(expert_class: Type[BaseExpert]):
        expert_registry.register(name, expert_class, category)
        return expert_class
    
    return decorator


def get_expert(name: str) -> Type[BaseExpert]:
    """Get expert class from global registry."""
    return expert_registry.get(name)


def create_expert(name: str, config: Any, **kwargs) -> BaseExpert:
    """Create expert instance from global registry."""
    return expert_registry.create(name, config, **kwargs)


def list_available_experts(category: Optional[str] = None) -> List[str]:
    """List available experts from global registry."""
    return expert_registry.list_experts(category)


def suggest_expert_combination(task_type: str, data_characteristics: Dict[str, Any]) -> List[str]:
    """Suggest expert combination from global registry."""
    return expert_registry.suggest_experts(task_type, data_characteristics)