"""
Model Builder for Modular Framework

This module provides a high-level interface for building complete models
from modular components with automatic registration and validation.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple, Type
from dataclasses import dataclass, field

from .base_interfaces import (
    BaseBackbone, BaseEmbedding, BaseAttention, BaseProcessor, 
    BaseFeedForward, BaseOutput, BaseComponent
)
from .config_schemas import (
    BackboneConfig, EmbeddingConfig, AttentionConfig,
    ComponentConfig
)
from .factory import ComponentFactory
from .registry import ComponentRegistry

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Complete model configuration"""
    # Core components
    backbone: Dict[str, Any] = field(default_factory=dict)
    embedding: Dict[str, Any] = field(default_factory=dict)
    attention: Dict[str, Any] = field(default_factory=dict)
    feedforward: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    
    # Optional components
    processor: Optional[Dict[str, Any]] = None
    adapters: Optional[List[Dict[str, Any]]] = None
    
    # Model settings
    model_name: str = "modular_model"
    task_type: str = "forecasting"
    
    # Training settings
    dropout: float = 0.1
    layer_norm: bool = True
    residual_connections: bool = True
    
    # Architecture settings
    d_model: int = 512
    num_layers: int = 6
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Set d_model in all component configs if not specified
        for component_config in [self.backbone, self.embedding, self.attention, 
                                self.feedforward, self.output]:
            if 'd_model' not in component_config:
                component_config['d_model'] = self.d_model
        
        # Validate required components
        if not self.backbone:
            raise ValueError("Backbone configuration is required")
        if not self.output:
            raise ValueError("Output configuration is required")


class ModularModel(nn.Module):
    """
    Complete modular model built from individual components
    
    This is the main model class that combines all modular components
    into a functional transformer-based model.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.task_type = config.task_type
        
        # Build components
        self._build_components()
        
        # Setup layer connections
        self._setup_connections()
        
        logger.info(f"ModularModel '{config.model_name}' initialized for {config.task_type}")
    
    def _build_components(self):
        """Build all model components"""
        factory = ComponentFactory.get_instance()
        
        # 1. Backbone (required)
        self.backbone = factory.create_backbone(
            self.config.backbone['type'],
            self.config.backbone
        )
        
        # 2. Embedding (optional - some backbones have built-in embedding)
        if self.config.embedding:
            self.embedding = factory.create_embedding(
                self.config.embedding['type'],
                self.config.embedding
            )
        else:
            self.embedding = None
        
        # 3. Attention (optional - some backbones handle attention)
        if self.config.attention:
            self.attention = factory.create_attention(
                self.config.attention['type'],
                self.config.attention
            )
        else:
            self.attention = None
        
        # 4. Feed-Forward (optional but recommended)
        if self.config.feedforward:
            self.feedforward = factory.create_feedforward(
                self.config.feedforward['type'],
                self.config.feedforward
            )
        else:
            self.feedforward = None
        
        # 5. Output head (required)
        self.output_head = factory.create_output(
            self.config.output['type'],
            self.config.output
        )
        
        # 6. Processor (optional)
        if self.config.processor:
            self.processor = factory.create_processor(
                self.config.processor['type'],
                self.config.processor
            )
        else:
            self.processor = None
        
        # 7. Adapters (optional)
        self.adapters = nn.ModuleList()
        if self.config.adapters:
            for adapter_config in self.config.adapters:
                adapter = factory.create_adapter(
                    adapter_config['type'],
                    adapter_config
                )
                self.adapters.append(adapter)
    
    def _setup_connections(self):
        """Setup layer normalization and residual connections"""
        d_model = self.config.d_model
        
        # Layer normalizations
        if self.config.layer_norm:
            self.backbone_norm = nn.LayerNorm(d_model)
            if self.attention:
                self.attention_norm = nn.LayerNorm(d_model)
            if self.feedforward:
                self.feedforward_norm = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(self.config.dropout)
    
    def forward(self, 
                x: torch.Tensor,
                x_mark: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the modular model
        
        Args:
            x: Input sequences [batch_size, seq_len, input_dim]
            x_mark: Time marks/covariates [batch_size, seq_len, mark_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Model outputs (format depends on output head type)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # 1. Apply adapters first (if any)
        for adapter in self.adapters:
            if hasattr(adapter, 'forward'):
                x = adapter(x, x_mark, attention_mask)
        
        # 2. Embedding (if separate from backbone)
        if self.embedding:
            x = self.embedding.embed_sequence(x, x_mark)
        
        # 3. Backbone processing
        if hasattr(self.backbone, 'forward'):
            # Modern backbone interface
            backbone_output = self.backbone.forward(x, attention_mask)
        else:
            # Legacy backbone interface
            backbone_output = self.backbone(x, attention_mask)
        
        # Apply backbone layer norm
        if hasattr(self, 'backbone_norm'):
            backbone_output = self.backbone_norm(backbone_output)
        
        # 4. Additional attention (if specified)
        if self.attention:
            # Self-attention with residual connection
            attn_output, _ = self.attention.apply_attention(
                backbone_output, backbone_output, backbone_output, attention_mask
            )
            
            if self.config.residual_connections:
                backbone_output = backbone_output + self.dropout(attn_output)
            else:
                backbone_output = attn_output
            
            if hasattr(self, 'attention_norm'):
                backbone_output = self.attention_norm(backbone_output)
        
        # 5. Feed-forward processing (if specified)
        if self.feedforward:
            ffn_output = self.feedforward.apply_feedforward(backbone_output)
            
            if self.config.residual_connections:
                backbone_output = backbone_output + self.dropout(ffn_output)
            else:
                backbone_output = ffn_output
            
            if hasattr(self, 'feedforward_norm'):
                backbone_output = self.feedforward_norm(backbone_output)
        
        # 6. Processor (if specified)
        if self.processor:
            processed_output = self.processor.process_sequence(
                embedded_input=x,
                backbone_output=backbone_output,
                target_length=kwargs.get('target_length', seq_len),
                **kwargs
            )
            backbone_output = processed_output
        
        # 7. Output head
        final_output = self.output_head.forward(backbone_output, **kwargs)
        
        return final_output
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'config': self.config,
            'components': {}
        }
        
        # Component information
        info['components']['backbone'] = {
            'type': self.backbone.get_backbone_type(),
            'capabilities': self.backbone.get_capabilities()
        }
        
        if self.embedding:
            info['components']['embedding'] = {
                'type': self.embedding.get_embedding_type(),
                'capabilities': self.embedding.get_capabilities()
            }
        
        if self.attention:
            info['components']['attention'] = {
                'type': self.attention.get_attention_type(),
                'capabilities': self.attention.get_capabilities()
            }
        
        if self.feedforward:
            info['components']['feedforward'] = {
                'type': self.feedforward.get_ffn_type(),
                'capabilities': self.feedforward.get_capabilities()
            }
        
        info['components']['output'] = {
            'type': self.output_head.get_output_type(),
            'capabilities': self.output_head.get_capabilities()
        }
        
        if self.processor:
            info['components']['processor'] = {
                'type': self.processor.get_processor_type(),
                'capabilities': self.processor.get_capabilities()
            }
        
        return info
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for each component"""
        counts = {}
        
        counts['backbone'] = sum(p.numel() for p in self.backbone.parameters())
        
        if self.embedding:
            counts['embedding'] = sum(p.numel() for p in self.embedding.parameters())
        
        if self.attention:
            counts['attention'] = sum(p.numel() for p in self.attention.parameters())
        
        if self.feedforward:
            counts['feedforward'] = sum(p.numel() for p in self.feedforward.parameters())
        
        counts['output'] = sum(p.numel() for p in self.output_head.parameters())
        
        if self.processor:
            counts['processor'] = sum(p.numel() for p in self.processor.parameters())
        
        for i, adapter in enumerate(self.adapters):
            counts[f'adapter_{i}'] = sum(p.numel() for p in adapter.parameters())
        
        counts['total'] = sum(counts.values())
        
        return counts


class ModelBuilder:
    """
    High-level builder for creating modular models
    
    Provides convenient methods for building common model architectures
    from modular components.
    """
    
    def __init__(self):
        self.factory = ComponentFactory()
        self.registry = ComponentRegistry()
    
    def build_model(self, config: ModelConfig) -> ModularModel:
        """Build a complete model from configuration"""
        # Validate configuration
        self._validate_config(config)
        
        # Create model
        model = ModularModel(config)
        
        logger.info(f"Built model '{config.model_name}' with {model.get_parameter_count()['total']:,} parameters")
        
        return model
    
    def build_forecasting_model(self,
                               backbone_type: str = "chronos",
                               horizon: int = 24,
                               d_model: int = 512,
                               **kwargs) -> ModularModel:
        """Build a forecasting model with sensible defaults"""
        
        config = ModelConfig(
            model_name=f"forecasting_{backbone_type}",
            task_type="forecasting",
            d_model=d_model,
            backbone={
                'type': backbone_type,
                'd_model': d_model,
                **kwargs.get('backbone_config', {})
            },
            feedforward={
                'type': 'standard_ffn',
                'd_model': d_model,
                'd_ff': d_model * 4,
                **kwargs.get('feedforward_config', {})
            },
            output={
                'type': 'forecasting',
                'd_model': d_model,
                'horizon': horizon,
                'output_dim': kwargs.get('output_dim', 1),
                **kwargs.get('output_config', {})
            }
        )
        
        return self.build_model(config)
    
    def build_classification_model(self,
                                  backbone_type: str = "bert",
                                  num_classes: int = 2,
                                  d_model: int = 512,
                                  **kwargs) -> ModularModel:
        """Build a classification model with sensible defaults"""
        
        config = ModelConfig(
            model_name=f"classification_{backbone_type}",
            task_type="classification",
            d_model=d_model,
            backbone={
                'type': backbone_type,
                'd_model': d_model,
                **kwargs.get('backbone_config', {})
            },
            feedforward={
                'type': 'standard_ffn',
                'd_model': d_model,
                'd_ff': d_model * 4,
                **kwargs.get('feedforward_config', {})
            },
            output={
                'type': 'classification',
                'd_model': d_model,
                'num_classes': num_classes,
                **kwargs.get('output_config', {})
            }
        )
        
        return self.build_model(config)
    
    def build_regression_model(self,
                              backbone_type: str = "simple_transformer",
                              output_dim: int = 1,
                              d_model: int = 512,
                              **kwargs) -> ModularModel:
        """Build a regression model with sensible defaults"""
        
        config = ModelConfig(
            model_name=f"regression_{backbone_type}",
            task_type="regression",
            d_model=d_model,
            backbone={
                'type': backbone_type,
                'd_model': d_model,
                **kwargs.get('backbone_config', {})
            },
            feedforward={
                'type': 'standard_ffn',
                'd_model': d_model,
                'd_ff': d_model * 4,
                **kwargs.get('feedforward_config', {})
            },
            output={
                'type': 'regression',
                'd_model': d_model,
                'output_dim': output_dim,
                **kwargs.get('output_config', {})
            }
        )
        
        return self.build_model(config)
    
    def _validate_config(self, config: ModelConfig):
        """Validate model configuration"""
        
        # Check if backbone type is registered
        backbone_type = config.backbone.get('type')
        if not self.registry.is_registered('backbone', backbone_type):
            available = self.registry.list_registered('backbone')
            raise ValueError(f"Unknown backbone type '{backbone_type}'. Available: {available}")
        
        # Check if output type is registered
        output_type = config.output.get('type')
        if not self.registry.is_registered('output', output_type):
            available = self.registry.list_registered('output')
            raise ValueError(f"Unknown output type '{output_type}'. Available: {available}")
        
        # Validate optional components
        if config.embedding:
            embedding_type = config.embedding.get('type')
            if not self.registry.is_registered('embedding', embedding_type):
                available = self.registry.list_registered('embedding')
                raise ValueError(f"Unknown embedding type '{embedding_type}'. Available: {available}")
        
        if config.attention:
            attention_type = config.attention.get('type')
            if not self.registry.is_registered('attention', attention_type):
                available = self.registry.list_registered('attention')
                raise ValueError(f"Unknown attention type '{attention_type}'. Available: {available}")
        
        if config.feedforward:
            ffn_type = config.feedforward.get('type')
            if not self.registry.is_registered('feedforward', ffn_type):
                available = self.registry.list_registered('feedforward')
                raise ValueError(f"Unknown feedforward type '{ffn_type}'. Available: {available}")
        
        logger.info("Model configuration validation passed")
    
    def list_available_components(self) -> Dict[str, List[str]]:
        """List all available component types"""
        return self.registry.list_components()
