"""
ModularComponent Base Class and Component Factory

This module implements the GCLI recommendations for a standardized component
interface and "dumb assembler" pattern for building modular autoformers.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass
from utils.logger import logger
from configs.schemas import (
    ModularAutoformerConfig, ComponentType, AttentionConfig, 
    DecompositionConfig, EncoderConfig, DecoderConfig, 
    SamplingConfig, OutputHeadConfig, LossConfig, BayesianConfig
)


@dataclass
class ComponentMetadata:
    """Metadata for component registration and validation"""
    name: str
    component_type: ComponentType
    required_params: List[str]
    optional_params: List[str]
    description: str
    dependencies: List[ComponentType] = None


class ModularComponent(ABC, nn.Module):
    """
    Base class for all modular components in the autoformer architecture
    
    This implements the GCLI recommendation for standardized component interfaces
    that enable clean composition and testing.
    """
    
    def __init__(self, config: Union[Dict[str, Any], ModularAutoformerConfig], **kwargs):
        super().__init__()
        self.component_type = None  # Should be set by subclasses
        self.metadata = None  # Should be set by subclasses
        self.config = config
        self._validated = False
        
        # Initialize component
        self._initialize_component(**kwargs)
        
    @abstractmethod
    def _initialize_component(self, **kwargs):
        """Initialize the component with given parameters"""
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass for the component"""
        pass
    
    def validate_config(self) -> bool:
        """Validate component configuration"""
        if self.metadata is None:
            return False
            
        # Check required parameters
        for param in self.metadata.required_params:
            if not hasattr(self, param) or getattr(self, param) is None:
                raise ValueError(f"Required parameter '{param}' not found in {self.metadata.name}")
        
        self._validated = True
        return True
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information for debugging and logging"""
        return {
            'name': self.metadata.name if self.metadata else 'Unknown',
            'type': self.component_type.value if self.component_type else 'Unknown',
            'validated': self._validated,
            'parameters': self.count_parameters(),
        }
    
    def count_parameters(self) -> int:
        """Count trainable parameters in this component"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionComponent(ModularComponent):
    """Base class for attention components"""
    
    def __init__(self, config: AttentionConfig, **kwargs):
        self.component_type = config.type
        super().__init__(config, **kwargs)


class DecompositionComponent(ModularComponent):
    """Base class for decomposition components"""
    
    def __init__(self, config: DecompositionConfig, **kwargs):
        self.component_type = config.type
        super().__init__(config, **kwargs)


class EncoderComponent(ModularComponent):
    """Base class for encoder components"""
    
    def __init__(self, config: EncoderConfig, **kwargs):
        self.component_type = config.type
        super().__init__(config, **kwargs)


class DecoderComponent(ModularComponent):
    """Base class for decoder components"""
    
    def __init__(self, config: DecoderConfig, **kwargs):
        self.component_type = config.type
        super().__init__(config, **kwargs)


class SamplingComponent(ModularComponent):
    """Base class for sampling components"""
    
    def __init__(self, config: SamplingConfig, **kwargs):
        self.component_type = config.type
        super().__init__(config, **kwargs)


class OutputHeadComponent(ModularComponent):
    """Base class for output head components"""
    
    def __init__(self, config: OutputHeadConfig, **kwargs):
        self.component_type = config.type
        super().__init__(config, **kwargs)


class LossComponent(ModularComponent):
    """Base class for loss components"""
    
    def __init__(self, config: LossConfig, **kwargs):
        self.component_type = config.type
        super().__init__(config, **kwargs)


class ComponentRegistry:
    """
    Registry for all modular components
    
    This implements the GCLI recommendation for centralized component management
    and enables the "dumb assembler" pattern.
    """
    
    def __init__(self):
        self._components: Dict[ComponentType, Type[ModularComponent]] = {}
        self._metadata: Dict[ComponentType, ComponentMetadata] = {}
        self._modifiers: Dict[str, callable] = {}
        
    def register_component(
        self, 
        component_type: ComponentType, 
        component_class: Type[ModularComponent],
        metadata: ComponentMetadata
    ):
        """Register a component with the registry"""
        self._components[component_type] = component_class
        self._metadata[component_type] = metadata
        
    def register_modifier(self, name: str, modifier_func: callable):
        """Register a component modifier (e.g., for Bayesian conversion)"""
        self._modifiers[name] = modifier_func
        
    def create_component(
        self, 
        component_type: ComponentType, 
        config: Union[Dict[str, Any], ModularAutoformerConfig],
        **kwargs
    ) -> ModularComponent:
        """Create a component instance"""
        if component_type not in self._components:
            raise ValueError(f"Component type {component_type} not registered")
            
        component_class = self._components[component_type]
        component = component_class(config, **kwargs)
        component.validate_config()
        
        return component
    
    def apply_modifier(
        self, 
        component: ModularComponent, 
        modifier_name: str, 
        **kwargs
    ) -> ModularComponent:
        """Apply a modifier to a component"""
        if modifier_name not in self._modifiers:
            raise ValueError(f"Modifier {modifier_name} not registered")
            
        modifier_func = self._modifiers[modifier_name]
        return modifier_func(component, **kwargs)
    
    def get_available_components(self) -> List[ComponentType]:
        """Get list of available component types"""
        return list(self._components.keys())
    
    def get_component_metadata(self, component_type: ComponentType) -> ComponentMetadata:
        """Get metadata for a component type"""
        return self._metadata.get(component_type)


class ModularAssembler:
    """
    "Dumb assembler" for building modular autoformers
    
    This implements the GCLI recommendation for a simple assembler that
    follows structured configuration without complex logic.
    """
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.components: Dict[str, ModularComponent] = {}
        
    def assemble_model(self, config: ModularAutoformerConfig) -> 'AssembledAutoformer':
        """
        Assemble a complete autoformer model from configuration
        
        This follows the "dumb assembler" pattern - it simply follows
        the configuration without making decisions.
        """
        
        # Create base components
        attention = self.registry.create_component(
            config.attention.type, 
            config.attention,
            d_model=config.d_model,
            seq_len=config.seq_len
        )
        
        decomposition = self.registry.create_component(
            config.decomposition.type,
            config.decomposition,
            d_model=config.d_model
        )
        
        encoder = self.registry.create_component(
            config.encoder.type,
            config.encoder,
            attention_comp=attention,
            decomp_comp=decomposition
        )
        
        decoder = self.registry.create_component(
            config.decoder.type,
            config.decoder,
            attention_comp=attention,
            decomp_comp=decomposition
        )
        
        sampling = self.registry.create_component(
            config.sampling.type,
            config.sampling,
            d_model=config.d_model
        )
        
        output_head = self.registry.create_component(
            config.output_head.type,
            config.output_head,
            d_model=config.d_model
        )
        
        loss_component = self.registry.create_component(
            config.loss.type,
            config.loss
        )
        
        # Apply Bayesian modifiers if needed
        components_to_modify = [encoder, decoder, output_head]
        if config.bayesian.enabled:
            modified_components = []
            for comp in components_to_modify:
                if any(layer_name in str(comp.__class__.__name__).lower() 
                       for layer_name in config.bayesian.layers_to_convert):
                    modified_comp = self.registry.apply_modifier(
                        comp, 
                        'bayesian',
                        prior_scale=config.bayesian.prior_scale,
                        posterior_scale_init=config.bayesian.posterior_scale_init
                    )
                    modified_components.append(modified_comp)
                else:
                    modified_components.append(comp)
            encoder, decoder, output_head = modified_components
        
        # Store components
        self.components = {
            'attention': attention,
            'decomposition': decomposition,
            'encoder': encoder,
            'decoder': decoder,
            'sampling': sampling,
            'output_head': output_head,
            'loss': loss_component
        }
        
        # Create assembled model
        assembled_model = AssembledAutoformer(
            config=config,
            components=self.components
        )
        
        return assembled_model
    
    def get_component_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of assembled components"""
        summary = {}
        for name, component in self.components.items():
            summary[name] = component.get_component_info()
        return summary


class AssembledAutoformer(nn.Module):
    """
    Complete autoformer model assembled from modular components
    
    This represents the final assembled model that follows the structured
    configuration and uses the "dumb assembler" pattern.
    """
    
    def __init__(self, config: ModularAutoformerConfig, components: Dict[str, ModularComponent]):
        super().__init__()
        self.config = config
        self.components = nn.ModuleDict(components)
        
        # Store component references for easy access
        self.attention = components['attention']
        self.decomposition = components['decomposition']
        self.encoder = components['encoder']
        self.decoder = components['decoder']
        self.sampling = components['sampling']
        self.output_head = components['output_head']
        self.loss_component = components['loss']
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass through the assembled model"""
        
        # For traditional autoformer architecture, we need to handle 
        # trend initialization and decomposition properly
        
        # Initialize trend (this would normally be done by the decoder)
        trend_init = torch.zeros(
            x_dec.shape[0], 
            x_dec.shape[1], 
            self.config.d_model, 
            device=x_dec.device
        )
        
        # Decompose decoder input for seasonal initialization
        if hasattr(self.decomposition, 'forward'):
            seasonal_init, _ = self.decomposition(x_dec)
        else:
            seasonal_init = x_dec
        
        # Encoder forward pass
        enc_out, _ = self.encoder(x_enc, mask)
        
        # Decoder forward pass
        if hasattr(self.decoder, 'forward') and len(self.decoder.forward.__code__.co_varnames) > 5:
            # Decoder expects trend parameter
            dec_out, trend_part = self.decoder(
                seasonal_init, enc_out, 
                x_mask=mask, cross_mask=None, 
                trend=trend_init
            )
        else:
            # Standard decoder
            dec_out = self.decoder(seasonal_init, enc_out, mask, None)
            trend_part = trend_init
        
        # Combine seasonal and trend if available
        if trend_part is not None and hasattr(self.config, 'combine_trend_seasonal') and self.config.combine_trend_seasonal:
            decoder_output = dec_out + trend_part
        else:
            decoder_output = dec_out
        
        # Output head
        output = self.output_head(decoder_output)
        
        return output
    
    def compute_loss(self, predictions, targets, **kwargs):
        """Compute loss using the loss component"""
        return self.loss_component(predictions, targets, **kwargs)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get complete model summary"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        component_summaries = {}
        for name, component in self.components.items():
            component_summaries[name] = component.get_component_info()
        
        return {
            'config': self.config.dict(),
            'total_parameters': total_params,
            'components': component_summaries,
            'model_type': 'AssembledAutoformer'
        }


# Global component registry instance
component_registry = ComponentRegistry()


def register_all_components():
    """Register all available components with the registry"""
    from configs.concrete_components import register_concrete_components
    register_concrete_components()
    logger.info("All GCLI components registered successfully")


# Bayesian modifier function
def bayesian_modifier(component: ModularComponent, **kwargs) -> ModularComponent:
    """
    Modifier to convert components to Bayesian variants
    
    This implements the GCLI recommendation for systematic handling
    of Bayesian components through modifiers.
    """
    try:
        from layers.bayesian_layers import make_bayesian
        
        # Apply Bayesian conversion
        bayesian_component = make_bayesian(
            component,
            prior_scale=kwargs.get('prior_scale', 1.0),
            posterior_scale_init=kwargs.get('posterior_scale_init', -3.0)
        )
        
        return bayesian_component
    except ImportError:
        logger.warning("Bayesian layers not available, returning original component")
        return component


# Register the Bayesian modifier
component_registry.register_modifier('bayesian', bayesian_modifier)
