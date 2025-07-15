"""
Concrete Backbone Implementations

This module provides concrete implementations of the BaseBackbone interface
for different language model architectures used in time series forecasting.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple

# Import transformers components with fallbacks
try:
    from transformers import AutoModel, AutoConfig, T5Config, T5EncoderModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModel = None
    AutoConfig = None
    T5Config = None
    T5EncoderModel = None

# Try to import Chronos-specific components (may not be available)
try:
    from transformers import ChronosTokenizer, ChronosConfig, ChronosForForecasting
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    ChronosTokenizer = None
    ChronosConfig = None
    ChronosForForecasting = None

from ..base_interfaces import BaseBackbone
from ..config_schemas import BackboneConfig

logger = logging.getLogger(__name__)


class ChronosBackbone(BaseBackbone):
    """
    Chronos-based backbone for time series forecasting
    
    Chronos is specifically designed for time series tasks and provides
    excellent performance for forecasting applications.
    """
    
    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        self.config = config
        self.model_name = getattr(config, 'model_name', 'amazon/chronos-t5-small')
        self.pretrained = getattr(config, 'pretrained', True)
        
        if not CHRONOS_AVAILABLE:
            logger.warning("Chronos components not available, falling back to standard T5")
            # Fall back to T5Backbone
            from .backbones import T5Backbone
            fallback_config = BackboneConfig(
                d_model=config.d_model,
                dropout=config.dropout,
                model_name='google/flan-t5-small'
            )
            self._fallback = T5Backbone(fallback_config)
            self._is_fallback = True
            return
        
        self._is_fallback = False
        
        try:
            # Initialize Chronos components
            self.tokenizer = ChronosTokenizer.from_pretrained(self.model_name)
            
            if self.pretrained:
                self.chronos_model = ChronosForForecasting.from_pretrained(self.model_name)
                self.chronos_config = self.chronos_model.config
            else:
                self.chronos_config = ChronosConfig.from_pretrained(self.model_name)
                self.chronos_model = ChronosForForecasting(self.chronos_config)
            
            # Extract the encoder
            self.encoder = self.chronos_model.model.encoder
            
            # Adaptation layers to match target dimensions
            encoder_hidden_size = self.chronos_config.d_model
            self.hidden_adaptation = None
            
            if config.d_model != encoder_hidden_size:
                self.hidden_adaptation = nn.Sequential(
                    nn.Linear(encoder_hidden_size, config.d_model),
                    nn.LayerNorm(config.d_model),
                    nn.Dropout(config.dropout)
                )
            
            logger.info(f"Chronos backbone initialized with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chronos: {e}")
            # Fall back to T5Backbone as last resort
            fallback_config = BackboneConfig(
                d_model=config.d_model,
                dropout=config.dropout,
                model_name='google/flan-t5-small'
            )
            from . import T5Backbone  # Import here to avoid circular imports
            self._fallback = T5Backbone(fallback_config)
            self._is_fallback = True
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Chronos backbone
        
        Args:
            input_ids: Tokenized input sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Hidden states [batch_size, seq_len, d_model]
        """
        if hasattr(self, '_is_fallback') and self._is_fallback:
            return self._fallback.forward(input_ids, attention_mask)
        
        try:
            # Get encoder outputs
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            hidden_states = encoder_outputs.last_hidden_state
            
            # Apply dimension adaptation if needed
            if self.hidden_adaptation is not None:
                hidden_states = self.hidden_adaptation(hidden_states)
            
            return hidden_states
            
        except Exception as e:
            logger.error(f"Error in Chronos backbone forward pass: {e}")
            raise
    
    def get_hidden_size(self) -> int:
        """Get the output hidden dimension"""
        return self.config.d_model
    
    def get_d_model(self) -> int:
        """Return the model dimension (d_model) of the backbone"""
        return self.config.d_model
    
    def supports_seq2seq(self) -> bool:
        """Return whether this backbone supports sequence-to-sequence processing"""
        return True  # Chronos supports seq2seq for forecasting
    
    def get_backbone_type(self) -> str:
        """Return the type identifier of this backbone"""
        return "chronos"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get backbone capabilities

        """
        return {
            'type': 'chronos',
            'pretrained': self.pretrained,
            'model_name': self.model_name,
            'supports_time_series': True,
            'supports_forecasting': True,
            'max_sequence_length': getattr(self.chronos_config, 'max_position_embeddings', 512)
        }


class T5Backbone(BaseBackbone):
    """
    T5-based backbone for sequence-to-sequence modeling
    
    T5 (Text-to-Text Transfer Transformer) provides strong performance
    for seq2seq tasks and can be adapted for time series applications.
    """
    
    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        self.config = config
        self.model_name = getattr(config, 'model_name', 't5-small')
        self.pretrained = getattr(config, 'pretrained', True)
        
        try:
            # Initialize T5 components
            if self.pretrained:
                self.t5_config = T5Config.from_pretrained(self.model_name)
                self.encoder = T5EncoderModel.from_pretrained(self.model_name)
            else:
                self.t5_config = T5Config.from_pretrained(self.model_name)
                self.encoder = T5EncoderModel(self.t5_config)
            
            # Adaptation layers
            encoder_hidden_size = self.t5_config.d_model
            self.hidden_adaptation = None
            
            if config.d_model != encoder_hidden_size:
                self.hidden_adaptation = nn.Sequential(
                    nn.Linear(encoder_hidden_size, config.d_model),
                    nn.LayerNorm(config.d_model),
                    nn.Dropout(config.dropout)
                )
            
            logger.info(f"T5 backbone initialized with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize T5 backbone: {e}")
            raise
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through T5 backbone
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Hidden states [batch_size, seq_len, d_model]
        """
        try:
            # Get encoder outputs
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            hidden_states = encoder_outputs.last_hidden_state
            
            # Apply dimension adaptation if needed
            if self.hidden_adaptation is not None:
                hidden_states = self.hidden_adaptation(hidden_states)
            
            return hidden_states
            
        except Exception as e:
            logger.error(f"Error in T5 backbone forward pass: {e}")
            raise
    
    def get_hidden_size(self) -> int:
        """Get the output hidden dimension"""
        return self.config.d_model
    
    def get_d_model(self) -> int:
        """Return the model dimension (d_model) of the backbone"""
        return self.config.d_model
    
    def supports_seq2seq(self) -> bool:
        """Return whether this backbone supports sequence-to-sequence processing"""
        return True  # T5 is designed for seq2seq
    
    def get_backbone_type(self) -> str:
        """Return the type identifier of this backbone"""
        return "t5"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get backbone capabilities"""
        return {
            'type': 't5',
            'pretrained': self.pretrained,
            'model_name': self.model_name,
            'supports_seq2seq': True,
            'max_sequence_length': getattr(self.t5_config, 'max_position_embeddings', 512)
        }


class BERTBackbone(BaseBackbone):
    """
    BERT-based backbone for representation learning
    
    BERT provides strong contextual representations and can be adapted
    for time series tasks with appropriate preprocessing.
    """
    
    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        self.config = config
        self.model_name = getattr(config, 'model_name', 'bert-base-uncased')
        self.pretrained = getattr(config, 'pretrained', True)
        
        try:
            # Initialize BERT components
            if self.pretrained:
                self.bert_config = AutoConfig.from_pretrained(self.model_name)
                self.encoder = AutoModel.from_pretrained(self.model_name)
            else:
                self.bert_config = AutoConfig.from_pretrained(self.model_name)
                self.encoder = AutoModel.from_config(self.bert_config)
            
            # Adaptation layers
            encoder_hidden_size = self.bert_config.hidden_size
            self.hidden_adaptation = None
            
            if config.d_model != encoder_hidden_size:
                self.hidden_adaptation = nn.Sequential(
                    nn.Linear(encoder_hidden_size, config.d_model),
                    nn.LayerNorm(config.d_model),
                    nn.Dropout(config.dropout)
                )
            
            logger.info(f"BERT backbone initialized with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize BERT backbone: {e}")
            raise
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through BERT backbone
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Hidden states [batch_size, seq_len, d_model]
        """
        try:
            # Get encoder outputs
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            hidden_states = outputs.last_hidden_state
            
            # Apply dimension adaptation if needed
            if self.hidden_adaptation is not None:
                hidden_states = self.hidden_adaptation(hidden_states)
            
            return hidden_states
            
        except Exception as e:
            logger.error(f"Error in BERT backbone forward pass: {e}")
            raise
    
    def get_hidden_size(self) -> int:
        """Get the output hidden dimension"""
        return self.config.d_model
    
    def get_d_model(self) -> int:
        """Return the model dimension (d_model) of the backbone"""
        return self.config.d_model
    
    def supports_seq2seq(self) -> bool:
        """Return whether this backbone supports sequence-to-sequence processing"""
        return False  # BERT is encoder-only, not seq2seq
    
    def get_backbone_type(self) -> str:
        """Return the type identifier of this backbone"""
        return "bert"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get backbone capabilities"""
        return {
            'type': 'bert',
            'pretrained': self.pretrained,
            'model_name': self.model_name,
            'supports_representation_learning': True,
            'max_sequence_length': getattr(self.bert_config, 'max_position_embeddings', 512)
        }


class SimpleTransformerBackbone(BaseBackbone):
    """
    Simple transformer backbone for lightweight applications
    
    A basic transformer implementation that can be used when
    pretrained models are not needed or available.
    """
    
    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        self.config = config
        
        # Build simple transformer
        self.num_layers = getattr(config, 'num_layers', 6)
        self.num_heads = getattr(config, 'num_heads', 8)
        self.dim_feedforward = getattr(config, 'dim_feedforward', 2048)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 1000, config.d_model) * 0.02
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        logger.info(f"Simple transformer backbone initialized with {self.num_layers} layers")
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through simple transformer
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Hidden states [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = embeddings.size()
        
        # Add positional encoding
        pos_embeddings = self.positional_encoding[:, :seq_len, :]
        embeddings = embeddings + pos_embeddings
        
        # Create attention mask for transformer
        src_key_padding_mask = None
        if attention_mask is not None:
            # Convert to padding mask (True for padding tokens)
            src_key_padding_mask = ~attention_mask.bool()
        
        # Apply transformer
        hidden_states = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        return hidden_states
    
    def get_hidden_size(self) -> int:
        """Get the output hidden dimension"""
        return self.config.d_model
    
    def get_d_model(self) -> int:
        """Return the model dimension (d_model) of the backbone"""
        return self.config.d_model
    
    def supports_seq2seq(self) -> bool:
        """Return whether this backbone supports sequence-to-sequence processing"""
        return False  # Simple transformer is encoder-only
    
    def get_backbone_type(self) -> str:
        """Return the type identifier of this backbone"""
        return "simple_transformer"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get backbone capabilities"""
        return {
            'type': 'simple_transformer',
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'lightweight': True,
            'trainable': True
        }
