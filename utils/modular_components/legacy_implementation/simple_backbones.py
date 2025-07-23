"""
Simplified Backbone Implementations with Robust Fallbacks

This module provides concrete implementations that work even when
specific libraries or models are not available.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple

from ..base_interfaces import BaseBackbone
from ..config_schemas import BackboneConfig

logger = logging.getLogger(__name__)


class SimpleTransformerBackbone(BaseBackbone):
    """
    Simple transformer backbone that works without external dependencies
    
    This is a fallback implementation that provides basic transformer
    functionality using only PyTorch.
    """
    
    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        self.nhead = getattr(config, 'nhead', 8)
        self.num_layers = getattr(config, 'num_layers', 6)
        self.dropout = config.dropout
        
        # Simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # Input/output projections
        self.input_projection = nn.Linear(getattr(config, 'input_dim', self.d_model), self.d_model)
        
        logger.info(f"Simple transformer backbone initialized with d_model={self.d_model}, layers={self.num_layers}")
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through simple transformer
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Hidden states [batch_size, seq_len, d_model]
        """
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Create attention mask if not provided
        if attention_mask is not None:
            # Convert to transformer format (True = masked)
            attention_mask = ~attention_mask.bool()
        
        # Apply transformer
        output = self.transformer(x, src_key_padding_mask=attention_mask)
        
        return output
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "SimpleTransformerBackbone",
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.nhead,
            "dropout": self.dropout
        }


class RobustHFBackbone(BaseBackbone):
    """
    Robust HuggingFace backbone with multiple fallback options
    
    This tries to use HF models but falls back gracefully if they're not available.
    """
    
    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        self.config = config
        self.model_name = getattr(config, 'model_name', 'google/flan-t5-small')
        self.d_model = config.d_model
        
        # Try different approaches in order of preference
        self.backend = None
        self.backend_type = None
        
        # 1. Try to use specified HF model
        if self._try_hf_model():
            logger.info(f"Using HuggingFace model: {self.model_name}")
        else:
            # 2. Fall back to simple transformer
            logger.warning("HuggingFace models not available, using simple transformer")
            self._init_simple_transformer()
    
    def _try_hf_model(self) -> bool:
        """Try to initialize HuggingFace model"""
        try:
            from transformers import AutoModel, AutoConfig
            
            # Try to load the model
            self.hf_model = AutoModel.from_pretrained(self.model_name)
            self.hf_config = self.hf_model.config
            
            # Get model dimension
            hf_d_model = getattr(self.hf_config, 'd_model', 
                               getattr(self.hf_config, 'hidden_size', self.d_model))
            
            # Add adaptation layer if needed
            if hf_d_model != self.d_model:
                self.adaptation = nn.Linear(hf_d_model, self.d_model)
            else:
                self.adaptation = None
            
            self.backend = self.hf_model
            self.backend_type = "huggingface"
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load HF model {self.model_name}: {e}")
            return False
    
    def _init_simple_transformer(self):
        """Initialize simple transformer fallback"""
        simple_config = BackboneConfig(
            d_model=self.d_model,
            dropout=self.config.dropout,
            nhead=8,
            num_layers=6,
            input_dim=getattr(self.config, 'input_dim', self.d_model)
        )
        
        self.backend = SimpleTransformerBackbone(simple_config)
        self.backend_type = "simple"
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with automatic backend selection
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim] or [batch_size, seq_len] for HF
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Hidden states [batch_size, seq_len, d_model]
        """
        if self.backend_type == "huggingface":
            # HuggingFace model expects different input format
            if x.dim() == 2:
                # Assume it's token IDs
                outputs = self.backend(input_ids=x, attention_mask=attention_mask)
            else:
                # Assume it's embeddings
                outputs = self.backend(inputs_embeds=x, attention_mask=attention_mask)
            
            hidden_states = outputs.last_hidden_state
            
            # Apply adaptation if needed
            if self.adaptation is not None:
                hidden_states = self.adaptation(hidden_states)
            
            return hidden_states
            
        else:
            # Simple transformer
            return self.backend.forward(x, attention_mask)
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_info(self) -> Dict[str, Any]:
        base_info = {
            "name": "RobustHFBackbone",
            "d_model": self.d_model,
            "backend_type": self.backend_type,
            "model_name": self.model_name
        }
        
        if hasattr(self.backend, 'get_info'):
            base_info.update(self.backend.get_info())
        
        return base_info


# Register the robust implementations
def get_robust_backbones():
    """Get dictionary of robust backbone implementations"""
    return {
        'simple_transformer': SimpleTransformerBackbone,
        'robust_hf': RobustHFBackbone,
        'simple': SimpleTransformerBackbone,  # Alias
        'auto': RobustHFBackbone,  # Alias for automatic selection
    }
