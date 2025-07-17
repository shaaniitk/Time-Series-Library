"""
Cross-Functionality Dependency Manager for Modular HF Architecture

This module provides comprehensive validation and management of dependencies
between modular components, ensuring compatibility and providing clear error
messages for incompatible combinations.
"""

import logging
from typing import Dict, List, Optional, Set, Any, Type, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch.nn as nn

from .base_interfaces import BaseComponent
from .registry import ComponentRegistry

logger = logging.getLogger(__name__)


@dataclass
class ComponentRequirement:
    """Defines a requirement between components"""
    source_component: str  # Component that has the requirement
    target_component: str  # Component type that must satisfy requirement
    capability_needed: str  # Specific capability needed
    is_optional: bool = False  # Whether this requirement is optional
    fallback_strategy: Optional[str] = None  # Strategy if requirement not met


@dataclass
class DimensionConstraint:
    """Defines dimensional compatibility constraints"""
    source_component: str
    source_attribute: str  # e.g., 'output_dim', 'd_model'
    target_component: str
    target_attribute: str
    relationship: str = 'equal'  # 'equal', 'divisible', 'multiple'
    adapter_allowed: bool = True  # Whether adapter can bridge mismatch


class ComponentMetadata:
    """Enhanced metadata for components with dependency information"""
    
    def __init__(self, component_class: Type[BaseComponent]):
        self.component_class = component_class
        self.capabilities: Set[str] = set()
        self.requirements: List[ComponentRequirement] = []
        self.dimension_constraints: List[DimensionConstraint] = []
        self.compatibility_tags: Set[str] = set()
        
        # Extract metadata from component class if available
        self._extract_component_metadata()
    
    def _extract_component_metadata(self):
        """Extract metadata from component class methods"""
        if hasattr(self.component_class, 'get_capabilities'):
            self.capabilities.update(self.component_class.get_capabilities())
        
        if hasattr(self.component_class, 'get_requirements'):
            reqs = self.component_class.get_requirements()
            for target_type, capability in reqs.items():
                self.requirements.append(
                    ComponentRequirement(
                        source_component=self.component_class.__name__,
                        target_component=target_type,
                        capability_needed=capability
                    )
                )
        
        if hasattr(self.component_class, 'get_compatibility_tags'):
            self.compatibility_tags.update(self.component_class.get_compatibility_tags())


class DependencyValidator:
    """Validates cross-functionality dependencies between components"""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.metadata_cache: Dict[str, ComponentMetadata] = {}
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
        # Built-in compatibility rules
        self.compatibility_rules = self._initialize_compatibility_rules()
        self.dimension_rules = self._initialize_dimension_rules()
    
    def _initialize_compatibility_rules(self) -> Dict[str, Dict[str, Set[str]]]:
        """Initialize built-in compatibility rules"""
        return {
            'processor': {
                'frequency_domain': {'autocorr_attention', 'fourier_attention'},
                'time_domain': {'multi_head_attention', 'sparse_attention'},
                'wavelet_domain': {'autocorr_attention', 'wavelet_attention'},
                'hierarchical': {'multi_scale_attention', 'hierarchical_attention'}
            },
            'loss': {
                'bayesian': {'bayesian_mse', 'bayesian_mae', 'bayesian_quantile'},
                'quantile': {'pinball_loss', 'quantile_loss', 'bayesian_quantile'},
                'hierarchical': {'hierarchical_loss', 'multi_scale_loss'}
            },
            'backbone': {
                'chronos': {'time_domain', 'frequency_domain'},
                'hf_transformer': {'time_domain', 'attention_based'},
                't5_encoder': {'seq2seq', 'attention_based'}
            }
        }
    
    def _initialize_dimension_rules(self) -> List[DimensionConstraint]:
        """Initialize built-in dimension compatibility rules"""
        return [
            DimensionConstraint('backbone', 'd_model', 'processor', 'input_dim'),
            DimensionConstraint('embedding', 'd_model', 'backbone', 'd_model'),
            DimensionConstraint('processor', 'output_dim', 'output', 'input_dim'),
            DimensionConstraint('backbone', 'd_model', 'attention', 'd_model'),
        ]
    
    def get_component_metadata(self, component_type: str, component_name: str) -> ComponentMetadata:
        """Get cached metadata for a component"""
        cache_key = f"{component_type}:{component_name}"
        
        if cache_key not in self.metadata_cache:
            component_class = self.registry.get(component_type, component_name)
            self.metadata_cache[cache_key] = ComponentMetadata(component_class)
        
        return self.metadata_cache[cache_key]
    
    def validate_configuration(self, config_dict: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a complete component configuration
        
        Args:
            config_dict: Dictionary with component configurations
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        # Extract component selections from config
        components = self._extract_component_selections(config_dict)
        
        # Run validation checks
        self._validate_component_requirements(components)
        self._validate_dimensional_compatibility(components, config_dict)
        self._validate_suite_specific_constraints(components, config_dict)
        self._validate_processing_strategy_compatibility(components)
        
        is_valid = len(self.validation_errors) == 0
        
        if is_valid:
            logger.info("✅ Configuration validation passed")
        else:
            logger.error(f"❌ Configuration validation failed with {len(self.validation_errors)} errors")
        
        return is_valid, self.validation_errors.copy(), self.validation_warnings.copy()
    
    def _extract_component_selections(self, config_dict: Dict[str, Any]) -> Dict[str, str]:
        """Extract component type -> component name mappings from config"""
        components = {}
        
        # Standard component mappings
        component_mappings = {
            'backbone_type': 'backbone',
            'processor_type': 'processor',
            'attention_type': 'attention',
            'loss_type': 'loss',
            'embedding_type': 'embedding',
            'output_type': 'output'
        }
        
        for config_key, component_type in component_mappings.items():
            if config_key in config_dict:
                components[component_type] = config_dict[config_key]
        
        return components
    
    def _validate_component_requirements(self, components: Dict[str, str]):
        """Validate that component requirements are satisfied"""
        for component_type, component_name in components.items():
            try:
                metadata = self.get_component_metadata(component_type, component_name)
                
                for requirement in metadata.requirements:
                    target_type = requirement.target_component
                    needed_capability = requirement.capability_needed
                    
                    if target_type not in components:
                        if not requirement.is_optional:
                            self.validation_errors.append(
                                f"Component {component_name} ({component_type}) requires "
                                f"a {target_type} component, but none configured"
                            )
                        continue
                    
                    target_name = components[target_type]
                    target_metadata = self.get_component_metadata(target_type, target_name)
                    
                    if needed_capability not in target_metadata.capabilities:
                        if requirement.fallback_strategy:
                            self.validation_warnings.append(
                                f"Component {component_name} requires capability "
                                f"'{needed_capability}' from {target_name}, using fallback: "
                                f"{requirement.fallback_strategy}"
                            )
                        else:
                            self.validation_errors.append(
                                f"Component {component_name} requires capability "
                                f"'{needed_capability}' from {target_name}, but it's not available. "
                                f"Available capabilities: {list(target_metadata.capabilities)}"
                            )
            
            except Exception as e:
                self.validation_errors.append(
                    f"Failed to validate requirements for {component_name}: {e}"
                )
    
    def _validate_dimensional_compatibility(self, components: Dict[str, str], config_dict: Dict[str, Any]):
        """Validate dimensional compatibility between components"""
        for constraint in self.dimension_rules:
            source_type = constraint.source_component
            target_type = constraint.target_component
            
            if source_type not in components or target_type not in components:
                continue  # Skip if components not present
            
            source_name = components[source_type]
            target_name = components[target_type]
            
            # Try to get dimensions from config
            source_dim = self._get_dimension_from_config(
                config_dict, source_type, constraint.source_attribute
            )
            target_dim = self._get_dimension_from_config(
                config_dict, target_type, constraint.target_attribute
            )
            
            if source_dim is None or target_dim is None:
                self.validation_warnings.append(
                    f"Could not determine dimensions for {source_name} -> {target_name} compatibility check"
                )
                continue
            
            # Check dimensional compatibility
            compatible = self._check_dimension_compatibility(
                source_dim, target_dim, constraint.relationship
            )
            
            if not compatible:
                error_msg = (
                    f"Dimension mismatch: {source_name}.{constraint.source_attribute} "
                    f"({source_dim}) incompatible with {target_name}.{constraint.target_attribute} "
                    f"({target_dim}) - relationship: {constraint.relationship}"
                )
                
                if constraint.adapter_allowed:
                    self.validation_warnings.append(
                        error_msg + " - Consider using an adapter layer"
                    )
                else:
                    self.validation_errors.append(error_msg)
    
    def _get_dimension_from_config(self, config_dict: Dict[str, Any], 
                                  component_type: str, attribute: str) -> Optional[int]:
        """Extract dimension value from configuration"""
        # Common dimension mappings
        if attribute == 'd_model':
            return config_dict.get('d_model', 512)
        elif attribute == 'input_dim':
            if component_type == 'processor':
                return config_dict.get('d_model', 512)
            elif component_type == 'output':
                return config_dict.get('d_model', 512)
        elif attribute == 'output_dim':
            if component_type == 'backbone':
                return config_dict.get('d_model', 512)
            elif component_type == 'processor':
                return config_dict.get('d_model', 512)
        
        return None
    
    def _check_dimension_compatibility(self, source_dim: int, target_dim: int, 
                                     relationship: str) -> bool:
        """Check if dimensions are compatible according to relationship"""
        if relationship == 'equal':
            return source_dim == target_dim
        elif relationship == 'divisible':
            return target_dim % source_dim == 0
        elif relationship == 'multiple':
            return source_dim % target_dim == 0
        else:
            logger.warning(f"Unknown dimension relationship: {relationship}")
            return True  # Default to compatible
    
    def _validate_suite_specific_constraints(self, components: Dict[str, str], 
                                           config_dict: Dict[str, Any]):
        """Validate constraints specific to model suites"""
        suite_name = config_dict.get('suite_name', '')
        
        if 'bayesian' in suite_name.lower():
            self._validate_bayesian_constraints(components)
        
        if 'quantile' in suite_name.lower():
            self._validate_quantile_constraints(components, config_dict)
        
        if 'hierarchical' in suite_name.lower():
            self._validate_hierarchical_constraints(components)
    
    def _validate_bayesian_constraints(self, components: Dict[str, str]):
        """Validate constraints for Bayesian models"""
        if 'loss' in components:
            loss_name = components['loss']
            loss_metadata = self.get_component_metadata('loss', loss_name)
            
            if 'bayesian' not in loss_metadata.capabilities:
                self.validation_errors.append(
                    f"Bayesian models require a loss function with 'bayesian' capability. "
                    f"Loss '{loss_name}' capabilities: {list(loss_metadata.capabilities)}"
                )
    
    def _validate_quantile_constraints(self, components: Dict[str, str], 
                                     config_dict: Dict[str, Any]):
        """Validate constraints for Quantile models"""
        if 'loss' in components:
            loss_name = components['loss']
            loss_metadata = self.get_component_metadata('loss', loss_name)
            
            if 'quantile' not in loss_metadata.capabilities:
                self.validation_warnings.append(
                    f"Quantile models typically require a loss function with 'quantile' capability. "
                    f"Loss '{loss_name}' capabilities: {list(loss_metadata.capabilities)}"
                )
        
        # Check for quantile levels configuration
        if 'quantile_levels' in config_dict:
            levels = config_dict['quantile_levels']
            if not isinstance(levels, list) or not all(0 < q < 1 for q in levels):
                self.validation_errors.append(
                    "Quantile levels must be a list of values between 0 and 1"
                )
    
    def _validate_hierarchical_constraints(self, components: Dict[str, str]):
        """Validate constraints for Hierarchical models"""
        if 'processor' in components:
            processor_name = components['processor']
            processor_metadata = self.get_component_metadata('processor', processor_name)
            
            if 'hierarchical' not in processor_metadata.capabilities:
                self.validation_warnings.append(
                    f"Hierarchical models typically require a processor with 'hierarchical' capability. "
                    f"Processor '{processor_name}' capabilities: {list(processor_metadata.capabilities)}"
                )
    
    def _validate_processing_strategy_compatibility(self, components: Dict[str, str]):
        """Validate compatibility between processing strategies and other components"""
        if 'processor' not in components:
            return
        
        processor_name = components['processor']
        
        # Check built-in compatibility rules
        for strategy, compatible_components in self.compatibility_rules.get('processor', {}).items():
            if strategy in processor_name.lower():
                # Check attention compatibility
                if 'attention' in components:
                    attention_name = components['attention']
                    if not any(comp in attention_name.lower() for comp in compatible_components):
                        self.validation_warnings.append(
                            f"Processor strategy '{strategy}' may not be optimal with "
                            f"attention '{attention_name}'. Consider: {list(compatible_components)}"
                        )
    
    def suggest_compatible_components(self, component_type: str, 
                                    existing_components: Dict[str, str]) -> List[str]:
        """Suggest compatible components based on existing selections"""
        suggestions = []
        available_components = self.registry.list_components(component_type)[component_type]
        
        for component_name in available_components:
            # Create a test configuration
            test_components = existing_components.copy()
            test_components[component_type] = component_name
            test_config = {f"{k}_type": v for k, v in test_components.items()}
            
            # Check if this would be valid
            is_valid, _, _ = self.validate_configuration(test_config)
            if is_valid:
                suggestions.append(component_name)
        
        return suggestions
    
    def get_adapter_suggestions(self, source_component: str, target_component: str,
                              source_dim: int, target_dim: int) -> Dict[str, Any]:
        """Suggest adapter configurations for dimension mismatches"""
        if source_dim == target_dim:
            return {'needed': False}
        
        return {
            'needed': True,
            'type': 'linear_adapter',
            'input_dim': source_dim,
            'output_dim': target_dim,
            'suggested_config': {
                'hidden_layers': [] if abs(source_dim - target_dim) < 128 else [max(source_dim, target_dim)],
                'activation': 'relu' if source_dim != target_dim else None,
                'dropout': 0.1 if abs(source_dim - target_dim) > 256 else 0.0
            }
        }
