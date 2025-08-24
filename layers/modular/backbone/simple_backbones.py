"""
Simplified Backbone Implementations with Robust Fallbacks

This module provides concrete implementations that work even when
specific libraries or models are not available.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple

from ...base_interfaces import BaseBackbone
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
    
    def get_d_model(self) -> int:
        return self.d_model
    
    def supports_seq2seq(self) -> bool:
        # Encoder-only fallback
        return False
    
    def get_backbone_type(self) -> str:
        return "simple_transformer"
    
    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Capabilities provided by this backbone."""
        return [
            "encoder_only",
            "lightweight",
            "no_external_dependencies",
            "deterministic",
            "robust_fallback",
            "forecasting_ready"
        ]

    @classmethod
    def get_compatibility_tags(cls) -> List[str]:
        return [
            "transformer_encoder",
            "time_series",
            "seq_len_preserving",
            "cpu_friendly",
            "low_memory"
        ]

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "d_model": "Model dimensionality (int, required)",
            "dropout": "Dropout rate (float, default=0.1)",
            "nhead": "Number of attention heads (int, default=8)",
            "num_layers": "Encoder layers (int, default=6)",
            "input_dim": "Input feature dimension before projection (int, default=d_model)"
        }

    @classmethod
    def recommend(cls, need_seq2seq: bool = False, offline: bool = True, low_resource: bool = True) -> bool:
        """Return whether this backbone is recommended under the provided constraints."""
        if need_seq2seq:
            return False  # Lacks decoder
        if offline or low_resource:
            return True
        return True  # Generally safe default
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through simple transformer
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
        info = {
            "name": "SimpleTransformerBackbone",
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.nhead,
            "dropout": self.dropout,
            "capabilities": self.get_capabilities(),
            "compatibility": self.get_compatibility_tags(),
            "usage_recommendations": {
                "ideal_scenarios": [
                    "Baseline benchmarking",
                    "CPU-only or offline environments",
                    "Fast iteration / prototyping",
                    "Deterministic reproducible runs"
                ],
                "avoid_if": [
                    "You require seq2seq generation",
                    "You need pretrained language semantics"
                ],
                "selection_logic": "Pick when environment is offline OR resources are constrained OR simplicity preferred"
            }
        }
        return info


class RobustHFBackbone(BaseBackbone):
    """
    Robust HuggingFace backbone with multiple fallback options
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
    
    def get_d_model(self) -> int:
        return self.d_model
    
    def supports_seq2seq(self) -> bool:
        return True if self.backend_type == "huggingface" else False
    
    def get_backbone_type(self) -> str:
        return "robust_hf"

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return [
            "adaptive",
            "seq2seq_optional",
            "pretrained_support",
            "scalable",
            "fallback_chain",
            "forecasting_ready"
        ]

    @classmethod
    def get_compatibility_tags(cls) -> List[str]:
        return [
            "transformer",
            "time_series",
            "seq2seq",
            "encoder_decoder_or_encoder",
            "gpu_acceleration_optional"
        ]

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "model_name": "HF model identifier (str, default=google/flan-t5-small)",
            "d_model": "Target internal model dimension (int, required)",
            "dropout": "Dropout rate (float, default=0.1)",
            "input_dim": "Input feature dimension when providing embeddings (int, default=d_model)"
        }

    @classmethod
    def recommend(cls, need_seq2seq: bool = True, offline: bool = False, low_resource: bool = False) -> bool:
        if offline:
            return False
        if low_resource and need_seq2seq:
            return True
        if need_seq2seq:
            return True
        return not low_resource
    
    def _try_hf_model(self) -> bool:
        try:
            from transformers import AutoModel, AutoConfig
            
            self.hf_model = AutoModel.from_pretrained(self.model_name)
            self.hf_config = self.hf_model.config
            
            hf_d_model = getattr(self.hf_config, 'd_model', 
                               getattr(self.hf_config, 'hidden_size', self.d_model))
            
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
        if self.backend_type == "huggingface":
            if x.dim() == 2:
                outputs = self.backend(input_ids=x, attention_mask=attention_mask)
            else:
                outputs = self.backend(inputs_embeds=x, attention_mask=attention_mask)
            
            hidden_states = outputs.last_hidden_state
            
            if self.adaptation is not None:
                hidden_states = self.adaptation(hidden_states)
            
            return hidden_states
            
        else:
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
        base_info.update({
            "capabilities": self.get_capabilities(),
            "compatibility": self.get_compatibility_tags(),
            "usage_recommendations": {
                "ideal_scenarios": [
                    "Need seq2seq generation",
                    "Leverage pretrained representations",
                    "Scalable forecasting with richer language/time context"
                ],
                "avoid_if": [
                    "Completely offline with no cached models",
                    "Ultra low-latency CPU micro-deployments"
                ],
                "selection_logic": "Choose when seq2seq or pretrained benefits outweigh resource cost; falls back automatically otherwise"
            }
        })
        
        return base_info


def get_robust_backbones():
    """Get dictionary of robust backbone implementations"""
    return {
        'simple_transformer': SimpleTransformerBackbone,
        'robust_hf': RobustHFBackbone,
        'simple': SimpleTransformerBackbone,
        'auto': RobustHFBackbone,
    }