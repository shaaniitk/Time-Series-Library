"""
Unified Registry for All Migrated Components
Provides a single interface to access all components migrated from layers/modular
"""

from typing import Dict, Any, Optional
import logging

from ..registry import ComponentRegistry
from ..config_schemas import ComponentConfig

logger = logging.getLogger(__name__)

# Import all migrated categories
from .attention_migrated import register_attention_components
from .decomposition_migrated import register_decomposition_components
from .encoder_migrated import register_encoder_components
from .decoder_migrated import register_decoder_components
from .sampling_migrated import register_sampling_components
from .output_heads_migrated import register_output_heads_components
from .losses_migrated import register_losses_components
from .layers_migrated import register_layers_components
from .fusion_migrated import register_fusion_components


class MigratedComponentRegistry:
    """Registry for all migrated components from layers/modular"""
    
    def __init__(self):
        self.registry = ComponentRegistry()
        self._register_all_migrated_components()
    
    def _register_all_migrated_components(self):
        """Register all migrated components"""
        logger.info("Registering all migrated components...")
        
        register_attention_components(self.registry)
        register_decomposition_components(self.registry)
        register_encoder_components(self.registry)
        register_decoder_components(self.registry)
        register_sampling_components(self.registry)
        register_output_heads_components(self.registry)
        register_losses_components(self.registry)
        register_layers_components(self.registry)
        register_fusion_components(self.registry)
        
        logger.info("All migrated components registered successfully")
    
    def get_component(self, category: str, name: str, config: ComponentConfig = None):
        """Get a component by category and name"""
        return self.registry.create(category, name, config)
    
    def list_components(self):
        """List all available components"""
        return self.registry.list_components()
    
    def get_migration_summary(self):
        """Get summary of migrated components"""
        return {
            "attention": 114,
            "decomposition": 25,
            "encoder": 21,
            "decoder": 17,
            "sampling": 18,
            "output_heads": 14,
            "losses": 55,
            "layers": 18,
            "fusion": 11,
        }

# Global instance
migrated_registry = MigratedComponentRegistry()
