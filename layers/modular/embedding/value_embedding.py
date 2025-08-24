import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Dict
import math

from ..base import BaseEmbedding
from configs.schemas import EmbeddingConfig

logger = logging.getLogger(__name__)

class ValueEmbedding(BaseEmbedding):
    """
    Value-based embedding for continuous time series values
    
    Converts continuous values to embedding space with optional
    binning and normalization strategies.
    """
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        self.num_features = getattr(config, 'num_features', 1)
        self.use_binning = getattr(config, 'use_binning', False)
        self.num_bins = getattr(config, 'num_bins', 100)
        
        if self.use_binning:
            # Discrete embedding for binned values
            self.value_embedding = nn.Embedding(self.num_bins, self.d_model)
        else:
            # Linear projection for continuous values
            self.value_projection = nn.Linear(self.num_features, self.d_model)
        
        # Additional layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        logger.info(f"Value embedding initialized: binning={self.use_binning}, features={self.num_features}")
    
    def forward(self, 
                values: torch.Tensor,
                value_ranges: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """
        Forward pass for value embedding
        
        Args:
            values: Input values [batch_size, seq_len, num_features]
            value_ranges: Optional (min, max) for normalization
            
        Returns:
            Value embeddings [batch_size, seq_len, d_model]
        """
        if self.use_binning:
            # Bin values and embed
            if value_ranges is not None:
                min_val, max_val = value_ranges
                normalized_values = (values - min_val) / (max_val - min_val)
            else:
                normalized_values = torch.sigmoid(values)  # Default normalization
            
            # Convert to bin indices
            bin_indices = (normalized_values * (self.num_bins - 1)).long()
            bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
            
            # Get embeddings for first feature only if multiple features
            if bin_indices.dim() > 2:
                bin_indices = bin_indices[..., 0]
            
            embeddings = self.value_embedding(bin_indices)
        else:
            # Project continuous values
            embeddings = self.value_projection(values)
        
        # Apply normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def embed_sequence(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Embed values sequence (alias to forward)."""
        return self.forward(values=x)
    
    def get_embedding_type(self) -> str:
        return "value"
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'type': 'value',
            'uses_binning': self.use_binning,
            'num_bins': self.num_bins,
            'num_features': self.num_features,
        }