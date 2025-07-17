"""
Abstract Base Interfaces for Modular Time Series Components

These interfaces define the contracts that all modular components must implement,
ensuring compatibility and interchangeability across different model architectures.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass


class BaseComponent(nn.Module, ABC):
    """
    Abstract base class for all modular components
    
    This provides the fundamental interface that all components must implement
    for compatibility across different model architectures.
    """
    
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self._info = {}
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass implementation - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the output dimension of this component"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get component information for logging and debugging"""
        base_info = {
            'component_type': self.__class__.__name__,
            'output_dim': self.get_output_dim(),
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        base_info.update(self._info)
        return base_info
    
    def set_info(self, key: str, value: Any):
        """Set component-specific information"""
        self._info[key] = value
    
    @classmethod
    def get_capabilities(cls) -> List[str]:
        """
        Return list of capabilities this component provides
        
        Examples: ['time_domain', 'frequency_domain', 'bayesian', 'quantile']
        """
        return []
    
    @classmethod  
    def get_requirements(cls) -> Dict[str, str]:
        """
        Return requirements this component has for other components
        
        Returns:
            Dict mapping component_type -> required_capability
            Example: {'attention': 'frequency_domain_compatible'}
        """
        return {}
    
    @classmethod
    def get_compatibility_tags(cls) -> List[str]:
        """
        Return compatibility tags for this component
        
        Examples: ['transformer_compatible', 'seq2seq', 'autoregressive']
        """
        return []
    
    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """
        Return configuration schema for this component
        
        Returns:
            Dict describing expected configuration parameters
        """
        return {}


class BaseBackbone(BaseComponent):
    """
    Abstract base class for backbone models (Chronos, T5, etc.)
    
    Backbone models provide the core transformer processing capability
    and can be swapped between different implementations.
    """
    
    @abstractmethod
    def get_d_model(self) -> int:
        """Return the model dimension (d_model) of the backbone"""
        pass
    
    @abstractmethod
    def supports_seq2seq(self) -> bool:
        """Return whether this backbone supports sequence-to-sequence processing"""
        pass
    
    @abstractmethod
    def get_backbone_type(self) -> str:
        """Return the type identifier of this backbone"""
        pass
    
    def get_output_dim(self) -> int:
        return self.get_d_model()


class BaseEmbedding(BaseComponent):
    """
    Abstract base class for embedding components
    
    Embedding components handle the conversion of input data and covariates
    into the model's representation space.
    """
    
    @abstractmethod
    def embed_sequence(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Embed a sequence with optional time marks/covariates
        
        Args:
            x: Input sequence [batch, seq_len, features]
            x_mark: Time marks/covariates [batch, seq_len, mark_features]
            
        Returns:
            Embedded sequence [batch, seq_len, d_model]
        """
        pass
    
    @abstractmethod
    def get_embedding_type(self) -> str:
        """Return the type identifier of this embedding"""
        pass


class BaseAttention(BaseComponent):
    """
    Abstract base class for attention mechanisms
    
    Attention components can be self-attention, cross-attention, autocorrelation,
    or any other attention-based mechanism.
    """
    
    @abstractmethod
    def apply_attention(self, 
                       queries: torch.Tensor, 
                       keys: torch.Tensor, 
                       values: torch.Tensor,
                       attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply attention mechanism
        
        Args:
            queries: Query tensor [batch, seq_len, d_model]
            keys: Key tensor [batch, seq_len, d_model]
            values: Value tensor [batch, seq_len, d_model]
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        pass
    
    @abstractmethod
    def get_attention_type(self) -> str:
        """Return the type identifier of this attention mechanism"""
        pass


class BaseProcessor(BaseComponent):
    """
    Abstract base class for processing strategies
    
    Processors define how the model processes sequences (seq2seq, encoder-only, etc.)
    and coordinate between different components.
    """
    
    @abstractmethod
    def process_sequence(self, 
                        embedded_input: torch.Tensor,
                        backbone_output: torch.Tensor,
                        target_length: int,
                        **kwargs) -> torch.Tensor:
        """
        Process sequence according to the strategy
        
        Args:
            embedded_input: Embedded input sequence
            backbone_output: Output from backbone model
            target_length: Desired output sequence length
            **kwargs: Additional processing parameters
            
        Returns:
            Processed sequence [batch, target_length, d_model]
        """
        pass
    
    @abstractmethod
    def get_processor_type(self) -> str:
        """Return the type identifier of this processor"""
        pass


class BaseFeedForward(BaseComponent):
    """
    Abstract base class for feed-forward networks
    
    Feed-forward components provide the final processing layer and can include
    standard FFN, mixture of experts, adaptive networks, etc.
    """
    
    @abstractmethod
    def apply_feedforward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward transformation
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Transformed tensor [batch, seq_len, output_dim]
        """
        pass
    
    @abstractmethod
    def get_ffn_type(self) -> str:
        """Return the type identifier of this feed-forward network"""
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the output dimension of this feed-forward network"""
        pass


class BaseLoss(BaseComponent):
    """
    Abstract base class for loss functions
    
    Loss components can include regression losses, uncertainty losses,
    multi-task losses, etc.
    """
    
    @abstractmethod
    def compute_loss(self, 
                    predictions: torch.Tensor, 
                    targets: torch.Tensor,
                    **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss between predictions and targets
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional loss parameters
            
        Returns:
            Loss tensor or dictionary of loss components
        """
        pass
    
    @abstractmethod
    def get_loss_type(self) -> str:
        """Return the type identifier of this loss function"""
        pass
    
    def get_output_dim(self) -> int:
        return 1  # Loss functions typically output scalar values


class BaseOutput(BaseComponent):
    """
    Abstract base class for output heads
    
    Output components handle the final projection and can include
    regression heads, uncertainty heads, multi-task heads, etc.
    """
    
    @abstractmethod
    def get_output_type(self) -> str:
        """Return the type identifier of this output head"""
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the output dimension of this output head"""
        pass


class BaseAdapter(BaseComponent):
    """
    Abstract base class for adapters
    
    Adapters modify or enhance existing components, such as adding
    covariate fusion, applying data transformations, or implementing
    domain adaptation strategies.
    """
    
    def __init__(self, wrapped_component: BaseComponent, adapter_config: Dict[str, Any]):
        super().__init__(adapter_config)
        self.wrapped_component = wrapped_component
        self.adapter_config = adapter_config
    
    @abstractmethod
    def get_adapter_type(self) -> str:
        """Return the type identifier of this adapter"""
        pass
    
    def get_output_dim(self) -> int:
        """Default: return the wrapped component's output dimension"""
        return self.wrapped_component.get_output_dim()
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get adapter capabilities including wrapped component info"""
        capabilities = {
            'adapter_type': self.get_adapter_type(),
            'wrapped_component': self.wrapped_component.__class__.__name__,
            'output_dim': self.get_output_dim()
        }
        if hasattr(self.wrapped_component, 'get_capabilities'):
            capabilities['wrapped_capabilities'] = self.wrapped_component.get_capabilities()
        return capabilities


# Type aliases for better code readability
ComponentType = Union[BaseBackbone, BaseEmbedding, BaseAttention, BaseProcessor, BaseFeedForward, BaseLoss, BaseOutput, BaseAdapter]
ConfigType = Union['BackboneConfig', 'EmbeddingConfig', 'AttentionConfig', 'ProcessorConfig', 'FeedForwardConfig', 'LossConfig', 'OutputConfig']
