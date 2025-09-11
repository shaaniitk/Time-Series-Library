from __future__ import annotations
from typing import Any, Optional, Dict, List
import torch
import torch.nn as nn
from ..base_interfaces import BaseBackbone
from ..config_schemas import BackboneConfig
from ..encoder import CrossformerEncoder
from utils.logger import logger

class CrossformerBackbone(BaseBackbone):
    """
    Crossformer-based backbone for time series forecasting.
    
    This backbone wraps the CrossformerEncoder to provide a complete
    backbone implementation that follows the modular architecture pattern.
    Crossformer is specifically designed for long-term time series forecasting
    with cross-dimension dependency modeling.
    """
    
    def __init__(self, config: BackboneConfig):
        super().__init__(config)
        self.config = config
        
        # Create a configuration object for the CrossformerEncoder
        encoder_config = type('EncoderConfig', (), {
            'enc_in': getattr(config, 'enc_in', 7),
            'seq_len': getattr(config, 'seq_len', 96),
            'd_model': config.d_model,
            'n_heads': getattr(config, 'n_heads', 8),
            'd_ff': getattr(config, 'd_ff', 2048),
            'e_layers': getattr(config, 'e_layers', 2),
            'dropout': config.dropout,
            'factor': getattr(config, 'factor', 3)
        })
        
        # Initialize the CrossformerEncoder
        self.encoder = CrossformerEncoder(encoder_config)
        
        # Output projection to ensure consistent output dimensions
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        
        logger.info(f"Crossformer backbone initialized with d_model={config.d_model}")
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Crossformer backbone.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, enc_in)
            attention_mask: Optional attention mask
            
        Returns:
            Hidden states [batch_size, seq_len, d_model]
        """
        logger.debug(f"CrossformerBackbone forward - input shape: {x.shape}")
        
        # Pass through the encoder
        encoder_output, _ = self.encoder(x, attention_mask)
        
        # Reshape encoder output to match expected backbone output format
        # encoder_output shape: (batch_size, enc_in, seg_num, d_model)
        # We need: (batch_size, seq_len, d_model)
        
        batch_size = encoder_output.shape[0]
        enc_in = encoder_output.shape[1]
        seg_num = encoder_output.shape[2]
        d_model = encoder_output.shape[3]
        
        # Flatten the segments and variables dimensions
        # This creates a sequence-like representation
        output = encoder_output.view(batch_size, enc_in * seg_num, d_model)
        
        # Apply output projection
        output = self.output_projection(output)
        
        logger.debug(f"CrossformerBackbone output shape: {output.shape}")
        
        return output
    
    def get_hidden_size(self) -> int:
        """Get the output hidden dimension"""
        return self.config.d_model
    
    def get_d_model(self) -> int:
        """Return the model dimension (d_model) of the backbone"""
        return self.config.d_model
    
    def supports_seq2seq(self) -> bool:
        """Return whether this backbone supports sequence-to-sequence processing"""
        return True  # Crossformer supports seq2seq for forecasting
    
    def get_backbone_type(self) -> str:
        """Return the type identifier of this backbone"""
        return "crossformer"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get backbone capabilities
        """
        return {
            'type': 'crossformer',
            'supports_time_series': True,
            'supports_forecasting': True,
            'supports_long_term': True,
            'cross_dimension_modeling': True,
            'max_sequence_length': getattr(self.config, 'seq_len', 96)
        }
    
    # ---------------- Recommendation / Metadata Extensions ---------------- #
    @classmethod
    def get_compatibility_tags(cls) -> List[str]:
        return [
            'transformer_encoder',
            'time_series',
            'forecasting',
            'long_term_forecasting',
            'cross_dimension',
            'seq2seq'
        ]
    
    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            'd_model': 'Internal model dimension (int, required)',
            'dropout': 'Dropout rate (float, default=0.1)',
            'enc_in': 'Number of input features (int, default=7)',
            'seq_len': 'Input sequence length (int, default=96)',
            'n_heads': 'Number of attention heads (int, default=8)',
            'd_ff': 'Feed-forward dimension (int, default=2048)',
            'e_layers': 'Number of encoder layers (int, default=2)',
            'factor': 'Attention factor (int, default=3)'
        }
    
    @classmethod
    def recommend(cls, need_forecasting: bool = True, need_long_term: bool = False, 
                 need_cross_dimension: bool = False) -> bool:
        """Return whether this backbone is recommended for given constraints."""
        if need_forecasting:
            return True
        if need_long_term or need_cross_dimension:
            return True
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get detailed information about this backbone instance"""
        return {
            'backbone_type': self.get_backbone_type(),
            'capabilities': self.get_capabilities(),
            'config': {
                'd_model': self.config.d_model,
                'dropout': self.config.dropout,
                'enc_in': getattr(self.config, 'enc_in', 7),
                'seq_len': getattr(self.config, 'seq_len', 96),
                'n_heads': getattr(self.config, 'n_heads', 8),
                'd_ff': getattr(self.config, 'd_ff', 2048),
                'e_layers': getattr(self.config, 'e_layers', 2),
                'factor': getattr(self.config, 'factor', 3)
            },
            'compatibility_tags': self.get_compatibility_tags(),
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class CrossformerBackboneWrapper:
    """
    Thin adapter for Crossformer backbone usable via unified registry.
    
    Automatically supplies minimal config attributes expected by the
    CrossformerBackbone implementation. This wrapper follows the same
    pattern as other backbone wrappers in the system.
    
    Parameters
    ----------
    d_model: int (default 512)
        Target internal model dimension.
    dropout: float (default 0.1)
        Dropout probability.
    enc_in: int (default 7)
        Number of input features.
    seq_len: int (default 96)
        Input sequence length.
    n_heads: int (default 8)
        Number of attention heads.
    d_ff: int (default 2048)
        Feed-forward dimension.
    e_layers: int (default 2)
        Number of encoder layers.
    factor: int (default 3)
        Attention factor.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        dropout: float = 0.1,
        enc_in: int = 7,
        seq_len: int = 96,
        n_heads: int = 8,
        d_ff: int = 2048,
        e_layers: int = 2,
        factor: int = 3,
        **kwargs: Any,
    ) -> None:
        cfg_dict = {
            'd_model': d_model,
            'dropout': dropout,
            'enc_in': enc_in,
            'seq_len': seq_len,
            'n_heads': n_heads,
            'd_ff': d_ff,
            'e_layers': e_layers,
            'factor': factor,
        }
        cfg_dict.update(kwargs)
        
        try:
            config = BackboneConfig(**cfg_dict)
            self._impl = CrossformerBackbone(config)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Crossformer backbone: {e}")
        
        self.d_model = d_model
    
    def forward(self, x, *args, **kwargs):
        return self._impl.forward(x, *args, **kwargs)
    
    def get_output_dim(self) -> int:
        """Compatibility hook"""
        return self.d_model