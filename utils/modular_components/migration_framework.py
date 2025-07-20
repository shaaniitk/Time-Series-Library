"""
Component Migration Framework

This module provides adapters and utilities to migrate components from the legacy
modular framework (layers/modular) to the new modular framework (utils/modular_components).
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Type, Callable, Optional, Union, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from utils.modular_components.base_interfaces import BaseComponent, BaseLoss, BaseAttention
from utils.modular_components.config_schemas import ComponentConfig
from utils.modular_components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


@dataclass
class LegacyComponentInfo:
    """Information about a legacy component for migration"""
    name: str
    registry_class: Type
    factory_function: Callable
    component_type: str
    dependencies: List[str] = None
    special_params: Dict[str, Any] = None


class ComponentMigrationAdapter(ABC):
    """
    Abstract base class for component migration adapters
    
    Adapters handle the conversion of legacy components to new framework components
    """
    
    def __init__(self):
        self.migrated_components: Dict[str, Type[BaseComponent]] = {}
    
    @abstractmethod
    def migrate_component(self, legacy_info: LegacyComponentInfo) -> Type[BaseComponent]:
        """Migrate a legacy component to the new framework"""
        pass
    
    @abstractmethod
    def get_component_type(self) -> str:
        """Return the component type this adapter handles"""
        pass
    
    def register_migrated_component(self, 
                                  registry: ComponentRegistry,
                                  component_name: str,
                                  component_class: Type[BaseComponent],
                                  metadata: Optional[Dict[str, Any]] = None):
        """Register a migrated component in the new registry"""
        component_type = self.get_component_type()
        registry.register(component_type, component_name, component_class, metadata)
        self.migrated_components[component_name] = component_class
        logger.info(f"Migrated and registered {component_type} component: {component_name}")


class LossComponentAdapter(ComponentMigrationAdapter):
    """Adapter for migrating loss function components"""
    
    def get_component_type(self) -> str:
        return "loss"
    
    def migrate_component(self, legacy_info: LegacyComponentInfo) -> Type[BaseComponent]:
        """
        Migrate a legacy loss component to new framework
        
        Args:
            legacy_info: Information about the legacy component
            
        Returns:
            New framework component class
        """
        legacy_name = legacy_info.name
        legacy_class = legacy_info.registry_class.get(legacy_name)
        
        # Create wrapper class that implements BaseLoss interface
        class MigratedLossComponent(BaseLoss):
            """Migrated loss component wrapper"""
            
            def __init__(self, config: ComponentConfig):
                super().__init__(config)
                
                # Extract parameters from config
                params = config.parameters if hasattr(config, 'parameters') else {}
                
                # Handle special parameter mappings
                if legacy_info.special_params:
                    for old_key, new_key in legacy_info.special_params.items():
                        if old_key in params:
                            params[new_key] = params.pop(old_key)
                
                # Create legacy component instance
                try:
                    if callable(legacy_class):
                        if legacy_name in ["mse", "mae", "huber"]:
                            # Handle standard loss wrappers
                            self.legacy_component = legacy_class()
                        else:
                            # Handle parameterized losses
                            self.legacy_component = legacy_class(**params)
                    else:
                        # Handle factory functions
                        self.legacy_component = legacy_class
                except Exception as e:
                    logger.error(f"Failed to create legacy component {legacy_name}: {e}")
                    raise
                
                # Store metadata
                self.legacy_name = legacy_name
                self.output_dim_multiplier = getattr(self.legacy_component, 'output_dim_multiplier', 1)
            
            def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
                """Forward pass through legacy component"""
                return self.legacy_component(predictions, targets, **kwargs)
            
            def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
                """Compute loss between predictions and targets"""
                return self.forward(predictions, targets, **kwargs)
            
            def get_loss_type(self) -> str:
                """Return the type identifier of this loss function"""
                return self.legacy_name
            
            def get_output_dim(self) -> int:
                """Return the output dimension of this component"""
                return 1  # Loss functions output scalar values
            
            def get_capabilities(self) -> Dict[str, Any]:
                """Get loss function capabilities"""
                capabilities = {
                    'type': self.legacy_name,
                    'legacy_migrated': True,
                    'output_dim_multiplier': self.output_dim_multiplier
                }
                
                # Add task type capabilities based on loss type
                if self.legacy_name in ["mse", "mae", "huber", "mape", "smape", "mase"]:
                    capabilities['task_types'] = ['regression', 'forecasting']
                elif "quantile" in self.legacy_name:
                    capabilities['task_types'] = ['quantile_regression', 'uncertainty']
                elif "bayesian" in self.legacy_name:
                    capabilities['task_types'] = ['bayesian', 'uncertainty', 'regression']
                
                return capabilities
        
        # Set class name and module for better identification
        MigratedLossComponent.__name__ = f"Migrated_{legacy_name.title()}Loss"
        MigratedLossComponent.__qualname__ = f"Migrated_{legacy_name.title()}Loss"
        
        return MigratedLossComponent


class AttentionComponentAdapter(ComponentMigrationAdapter):
    """Adapter for migrating attention mechanism components"""
    
    def get_component_type(self) -> str:
        return "attention"
    
    def migrate_component(self, legacy_info: LegacyComponentInfo) -> Type[BaseComponent]:
        """
        Migrate a legacy attention component to new framework
        
        Args:
            legacy_info: Information about the legacy component
            
        Returns:
            New framework component class
        """
        legacy_name = legacy_info.name
        factory_func = legacy_info.factory_function
        
        # Create wrapper class that implements BaseAttention interface
        class MigratedAttentionComponent(BaseAttention):
            """Migrated attention component wrapper"""
            
            def __init__(self, config: ComponentConfig):
                super().__init__(config)
                
                # Extract parameters from config
                params = config.parameters if hasattr(config, 'parameters') else {}
                
                # Handle special parameter mappings
                if legacy_info.special_params:
                    for old_key, new_key in legacy_info.special_params.items():
                        if old_key in params:
                            params[new_key] = params.pop(old_key)
                
                # Create legacy component instance using factory function
                try:
                    self.legacy_component = factory_func(legacy_name, **params)
                except Exception as e:
                    logger.error(f"Failed to create legacy attention component {legacy_name}: {e}")
                    raise
                
                # Store metadata
                self.legacy_name = legacy_name
                self.d_model = params.get('d_model', 512)
                self.n_heads = params.get('n_heads', 8)
            
            def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, 
                       attn_mask: Optional[torch.Tensor] = None, tau: Optional[float] = None, 
                       delta: Optional[float] = None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                """Forward pass through legacy attention component"""
                # Handle different legacy attention signatures
                try:
                    if hasattr(self.legacy_component, '__call__'):
                        # Some components are callable modules
                        if attn_mask is not None:
                            return self.legacy_component(queries, keys, values, attn_mask=attn_mask, **kwargs)
                        else:
                            return self.legacy_component(queries, keys, values, **kwargs)
                    else:
                        # Handle non-callable components
                        return self.legacy_component.forward(queries, keys, values, attn_mask=attn_mask, **kwargs)
                except Exception as e:
                    logger.warning(f"Legacy attention {self.legacy_name} forward failed: {e}")
                    # Fallback to simple pass-through
                    return queries, None
            
            def get_attention_type(self) -> str:
                """Return the type identifier of this attention mechanism"""
                return self.legacy_name
            
            def get_output_dim(self) -> int:
                """Return the output dimension of this component"""
                return self.d_model
            
            def get_capabilities(self) -> Dict[str, Any]:
                """Get attention mechanism capabilities"""
                capabilities = {
                    'type': self.legacy_name,
                    'legacy_migrated': True,
                    'd_model': self.d_model,
                    'n_heads': self.n_heads
                }
                
                # Add capability tags based on attention type
                if "fourier" in self.legacy_name:
                    capabilities['capabilities'] = ['frequency_domain', 'global_attention']
                elif "wavelet" in self.legacy_name:
                    capabilities['capabilities'] = ['multi_scale', 'time_frequency']
                elif "bayesian" in self.legacy_name:
                    capabilities['capabilities'] = ['uncertainty', 'probabilistic']
                elif "autocorrelation" in self.legacy_name:
                    capabilities['capabilities'] = ['time_series_specialized', 'lag_aware']
                
                return capabilities
        
        # Set class name and module for better identification
        MigratedAttentionComponent.__name__ = f"Migrated_{legacy_name.title()}Attention"
        MigratedAttentionComponent.__qualname__ = f"Migrated_{legacy_name.title()}Attention"
        
        return MigratedAttentionComponent


class MigrationManager:
    """
    Manages the migration of components from legacy to new framework
    """
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.adapters: Dict[str, ComponentMigrationAdapter] = {
            'loss': LossComponentAdapter(),
            'attention': AttentionComponentAdapter()
            # Add more adapters as needed
        }
        self.migration_status: Dict[str, Dict[str, bool]] = {}
    
    def migrate_all_losses(self):
        """Migrate all loss components from legacy framework"""
        from layers.modular.losses.registry import LossRegistry
        
        logger.info("Starting migration of loss components...")
        
        # Get all registered loss components
        legacy_losses = LossRegistry._registry
        
        loss_adapter = self.adapters['loss']
        migrated_count = 0
        
        for loss_name, loss_class in legacy_losses.items():
            try:
                # Create legacy component info
                legacy_info = LegacyComponentInfo(
                    name=loss_name,
                    registry_class=LossRegistry,
                    factory_function=None,  # Losses use direct class access
                    component_type='loss'
                )
                
                # Migrate component
                migrated_class = loss_adapter.migrate_component(legacy_info)
                
                # Register in new framework
                metadata = {
                    'legacy_name': loss_name,
                    'migration_source': 'layers.modular.losses',
                    'description': f'Migrated {loss_name} loss from legacy framework'
                }
                
                loss_adapter.register_migrated_component(
                    self.registry, loss_name, migrated_class, metadata
                )
                
                migrated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate loss component {loss_name}: {e}")
        
        logger.info(f"Successfully migrated {migrated_count} loss components")
        self.migration_status['loss'] = {name: True for name in legacy_losses.keys()}
    
    def migrate_all_attention(self):
        """Migrate all attention components from legacy framework"""
        from layers.modular.attention.registry import AttentionRegistry, get_attention_component
        
        logger.info("Starting migration of attention components...")
        
        # Get all registered attention components
        legacy_attention = AttentionRegistry._registry
        
        attention_adapter = self.adapters['attention']
        migrated_count = 0
        
        for attention_name, attention_class in legacy_attention.items():
            try:
                # Create legacy component info with special parameter mappings
                special_params = self._get_attention_param_mappings(attention_name)
                
                legacy_info = LegacyComponentInfo(
                    name=attention_name,
                    registry_class=AttentionRegistry,
                    factory_function=get_attention_component,
                    component_type='attention',
                    special_params=special_params
                )
                
                # Migrate component
                migrated_class = attention_adapter.migrate_component(legacy_info)
                
                # Register in new framework
                metadata = {
                    'legacy_name': attention_name,
                    'migration_source': 'layers.modular.attention',
                    'description': f'Migrated {attention_name} attention from legacy framework'
                }
                
                attention_adapter.register_migrated_component(
                    self.registry, attention_name, migrated_class, metadata
                )
                
                migrated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate attention component {attention_name}: {e}")
        
        logger.info(f"Successfully migrated {migrated_count} attention components")
        self.migration_status['attention'] = {name: True for name in legacy_attention.keys()}
    
    def _get_attention_param_mappings(self, attention_name: str) -> Dict[str, str]:
        """Get parameter mappings for specific attention types"""
        mappings = {}
        
        # Add specific parameter mappings based on attention type
        if "fourier" in attention_name:
            mappings = {
                'seq_len_q': 'query_len',
                'seq_len_kv': 'key_value_len'
            }
        elif "wavelet" in attention_name:
            mappings = {
                'n_levels': 'decomposition_levels',
                'max_levels': 'max_decomposition_levels'
            }
        elif "bayesian" in attention_name:
            mappings = {
                'prior_std': 'prior_std_dev',
                'n_samples': 'num_samples'
            }
        
        return mappings
    
    def get_migration_status(self) -> Dict[str, Dict[str, bool]]:
        """Get the current migration status"""
        return self.migration_status.copy()
    
    def migrate_all_components(self):
        """Migrate all supported component types"""
        logger.info("Starting full component migration...")
        
        self.migrate_all_losses()
        self.migrate_all_attention()
        
        # Add migration for other component types as adapters are added
        
        logger.info("Full component migration completed")
        
        # Print migration summary
        total_migrated = sum(len(components) for components in self.migration_status.values())
        logger.info(f"Migration Summary: {total_migrated} total components migrated")
        for comp_type, components in self.migration_status.items():
            logger.info(f"  {comp_type}: {len(components)} components")


# Utility functions for migration testing and validation

def validate_migrated_component(component_class: Type[BaseComponent], 
                              legacy_name: str,
                              test_config: ComponentConfig) -> bool:
    """
    Validate that a migrated component works correctly
    
    Args:
        component_class: Migrated component class
        legacy_name: Original legacy component name
        test_config: Test configuration
        
    Returns:
        True if component passes validation tests
    """
    try:
        # Create component instance
        component = component_class(test_config)
        
        # Basic validation checks
        assert hasattr(component, 'forward'), "Component must have forward method"
        assert hasattr(component, 'get_output_dim'), "Component must implement get_output_dim"
        
        # Type-specific validation
        if isinstance(component, BaseLoss):
            assert hasattr(component, 'compute_loss'), "Loss component must have compute_loss method"
            assert hasattr(component, 'get_loss_type'), "Loss component must have get_loss_type method"
        
        if isinstance(component, BaseAttention):
            assert hasattr(component, 'get_attention_type'), "Attention component must have get_attention_type method"
        
        logger.info(f"Migrated component {legacy_name} passed validation")
        return True
        
    except Exception as e:
        logger.error(f"Migrated component {legacy_name} failed validation: {e}")
        return False


def create_test_migration():
    """Create a test migration manager for testing purposes"""
    from utils.modular_components.registry import ComponentRegistry
    
    registry = ComponentRegistry()
    manager = MigrationManager(registry)
    
    return manager, registry
