"""
Factory Classes for Component Creation

These factories provide a clean interface for creating components with
automatic type checking, validation, and fallback mechanisms.
"""

import logging
from typing import Dict, Type, Any, Optional, Union
from .base_interfaces import (
    BaseComponent, BaseBackbone, BaseEmbedding, BaseAttention,
    BaseProcessor, BaseFeedForward, BaseLoss, BaseOutput
)
from .config_schemas import (
    ComponentConfig, BackboneConfig, EmbeddingConfig, AttentionConfig,
    ProcessorConfig, FeedForwardConfig, LossConfig, OutputConfig
)
from .registry import get_global_registry

logger = logging.getLogger(__name__)


class ComponentFactory:
    """
    Base factory class for creating components
    
    Provides common functionality for all component factories including
    validation, error handling, and fallback mechanisms.
    """
    
    def __init__(self, component_type: str, base_class: Type[BaseComponent]):
        self.component_type = component_type
        self.base_class = base_class
        self.registry = get_global_registry()
    
    def create(self, component_name: str, config: ComponentConfig) -> BaseComponent:
        """
        Create a component instance
        
        Args:
            component_name: Name of the component to create
            config: Configuration for the component
            
        Returns:
            Instantiated component
        """
        try:
            # Validate config type
            self._validate_config(config)
            
            # Create component
            component = self.registry.create(self.component_type, component_name, config)
            
            # Validate component type
            if not isinstance(component, self.base_class):
                raise TypeError(f"Component {component_name} does not inherit from {self.base_class.__name__}")
            
            logger.info(f"Successfully created {self.component_type} component: {component_name}")
            return component
            
        except Exception as e:
            logger.error(f"Failed to create {self.component_type} component '{component_name}': {e}")
            return self._handle_creation_error(component_name, config, e)
    
    def _validate_config(self, config: ComponentConfig):
        """Validate the configuration object"""
        if not isinstance(config, ComponentConfig):
            raise TypeError(f"Config must be a ComponentConfig instance, got {type(config)}")
    
    def _handle_creation_error(self, component_name: str, config: ComponentConfig, error: Exception) -> BaseComponent:
        """
        Handle component creation errors with fallback mechanisms
        
        Args:
            component_name: Name of the failed component
            config: Configuration that failed
            error: The exception that occurred
            
        Returns:
            Fallback component or re-raises the error
        """
        # Default behavior: re-raise the error
        # Subclasses can override this for specific fallback logic
        raise error
    
    def list_available(self) -> list:
        """List all available components of this type"""
        components = self.registry.list_components(self.component_type)
        return components.get(self.component_type, [])
    
    def is_available(self, component_name: str) -> bool:
        """Check if a component is available"""
        return self.registry.is_registered(self.component_type, component_name)


class BackboneFactory(ComponentFactory):
    """Factory for creating backbone components"""
    
    def __init__(self):
        super().__init__("backbone", BaseBackbone)
    
    def create(self, backbone_type: str, config: BackboneConfig) -> BaseBackbone:
        """
        Create a backbone component with fallback support
        
        Args:
            backbone_type: Type of backbone to create
            config: Backbone configuration
            
        Returns:
            Backbone component instance
        """
        try:
            return super().create(backbone_type, config)
        except Exception as e:
            logger.warning(f"Failed to create backbone '{backbone_type}': {e}")
            return self._try_fallback_backbones(config, e)
    
    def _try_fallback_backbones(self, config: BackboneConfig, original_error: Exception) -> BaseBackbone:
        """Try fallback backbone models"""
        for fallback_model in config.fallback_models:
            try:
                logger.info(f"Trying fallback backbone: {fallback_model}")
                fallback_config = BackboneConfig(
                    backbone_type=fallback_model,
                    d_model=config.d_model,
                    dropout=config.dropout
                )
                return super().create(fallback_model, fallback_config)
            except Exception as e:
                logger.warning(f"Fallback backbone '{fallback_model}' also failed: {e}")
                continue
        
        # If all fallbacks fail, raise the original error
        raise original_error


class EmbeddingFactory(ComponentFactory):
    """Factory for creating embedding components"""
    
    def __init__(self):
        super().__init__("embedding", BaseEmbedding)
    
    def create(self, embedding_type: str, config: EmbeddingConfig) -> BaseEmbedding:
        """Create an embedding component"""
        return super().create(embedding_type, config)


class AttentionFactory(ComponentFactory):
    """Factory for creating attention components"""
    
    def __init__(self):
        super().__init__("attention", BaseAttention)
    
    def create(self, attention_type: str, config: AttentionConfig) -> BaseAttention:
        """Create an attention component"""
        return super().create(attention_type, config)


class ProcessorFactory(ComponentFactory):
    """Factory for creating processor components"""
    
    def __init__(self):
        super().__init__("processor", BaseProcessor)
    
    def create(self, processor_type: str, config: ProcessorConfig) -> BaseProcessor:
        """Create a processor component"""
        return super().create(processor_type, config)


class FeedForwardFactory(ComponentFactory):
    """Factory for creating feed-forward components"""
    
    def __init__(self):
        super().__init__("feedforward", BaseFeedForward)
    
    def create(self, ffn_type: str, config: FeedForwardConfig) -> BaseFeedForward:
        """Create a feed-forward component"""
        return super().create(ffn_type, config)


class LossFactory(ComponentFactory):
    """Factory for creating loss components"""
    
    def __init__(self):
        super().__init__("loss", BaseLoss)
    
    def create(self, loss_type: str, config: LossConfig) -> BaseLoss:
        """Create a loss component"""
        return super().create(loss_type, config)


class OutputFactory(ComponentFactory):
    """Factory for creating output components"""
    
    def __init__(self):
        super().__init__("output", BaseOutput)
    
    def create(self, output_type: str, config: OutputConfig) -> BaseOutput:
        """Create an output component"""
        return super().create(output_type, config)


# Global factory instances for convenience
backbone_factory = BackboneFactory()
embedding_factory = EmbeddingFactory()
attention_factory = AttentionFactory()
processor_factory = ProcessorFactory()
feedforward_factory = FeedForwardFactory()
loss_factory = LossFactory()
output_factory = OutputFactory()


# Convenience functions
def create_backbone(backbone_type: str, config: BackboneConfig) -> BaseBackbone:
    """Create a backbone component"""
    return backbone_factory.create(backbone_type, config)


def create_embedding(embedding_type: str, config: EmbeddingConfig) -> BaseEmbedding:
    """Create an embedding component"""
    return embedding_factory.create(embedding_type, config)


def create_attention(attention_type: str, config: AttentionConfig) -> BaseAttention:
    """Create an attention component"""
    return attention_factory.create(attention_type, config)


def create_processor(processor_type: str, config: ProcessorConfig) -> BaseProcessor:
    """Create a processor component"""
    return processor_factory.create(processor_type, config)


def create_feedforward(ffn_type: str, config: FeedForwardConfig) -> BaseFeedForward:
    """Create a feed-forward component"""
    return feedforward_factory.create(ffn_type, config)


def create_loss(loss_type: str, config: LossConfig) -> BaseLoss:
    """Create a loss component"""
    return loss_factory.create(loss_type, config)


def create_output(output_type: str, config: OutputConfig) -> BaseOutput:
    """Create an output component"""
    return output_factory.create(output_type, config)
