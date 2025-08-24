import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from ..base import BaseEmbedding
from ..core.configs import EmbeddingConfig
from .temporal_embedding import TemporalEmbedding
from .value_embedding import ValueEmbedding
from .covariate_embedding import CovariateEmbedding

logger = logging.getLogger(__name__)

class HybridEmbedding(BaseEmbedding):
    """
    Hybrid embedding combining temporal, value, and covariate embeddings
    
    Flexible combination of multiple embedding strategies with configurable
    fusion mechanisms.
    """
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        
        # Component embeddings
        self.temporal_embedding = TemporalEmbedding(config)
        self.value_embedding = ValueEmbedding(config)
        self.covariate_embedding = CovariateEmbedding(config)
        
        # Combination strategy
        self.combination_strategy = getattr(config, 'combination_strategy', 'concat')
        self.combination_weights = None
        
        if self.combination_strategy == 'weighted':
            self.combination_weights = nn.Parameter(
                torch.ones(3) / 3  # Assuming 3 components
            )
        elif self.combination_strategy == 'concat':
            concat_dim = (
                self.temporal_embedding.get_output_dim() +
                self.value_embedding.get_output_dim() +
                self.covariate_embedding.get_output_dim()
            )
            self.projection = nn.Linear(concat_dim, self.d_model)
        else:
            # 'add' or others - assume same dimension
            if not (self.temporal_embedding.get_output_dim() ==
                    self.value_embedding.get_output_dim() ==
                    self.covariate_embedding.get_output_dim()):
                raise ValueError("All embeddings must have same dimension for 'add' strategy")
        
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        logger.info(f"Hybrid embedding initialized with strategy: {self.combination_strategy}")
    
    def forward(self, 
                values: torch.Tensor,
                temporal_marks: Optional[torch.Tensor] = None,
                categorical_covariates: Optional[Dict[str, torch.Tensor]] = None,
                numerical_covariates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for hybrid embedding
        
        Args:
            values: Input values [batch_size, seq_len, num_features]
            temporal_marks: Temporal features [batch_size, seq_len, temp_features]
            categorical_covariates: Dict of categorical covariates
            numerical_covariates: Numerical covariates [batch_size, seq_len, num_covariates]
            
        Returns:
            Hybrid embeddings [batch_size, seq_len, d_model]
        """
        # Get individual embeddings
        temp_emb = self.temporal_embedding(temporal_marks) if temporal_marks is not None else 0
        value_emb = self.value_embedding(values)
        cov_emb = self.covariate_embedding(
            categorical_data=categorical_covariates,
            numerical_data=numerical_covariates
        )
        
        # Combine embeddings
        if self.combination_strategy == 'add':
            combined = temp_emb + value_emb + cov_emb
        elif self.combination_strategy == 'concat':
            combined = torch.cat([temp_emb, value_emb, cov_emb], dim=-1)
            combined = self.projection(combined)
        elif self.combination_strategy == 'weighted':
            weights = torch.softmax(self.combination_weights, dim=0)
            combined = (
                weights[0] * temp_emb +
                weights[1] * value_emb +
                weights[2] * cov_emb
            )
        else:
            raise ValueError(f"Unknown combination strategy: {self.combination_strategy}")
        
        # Apply normalization and dropout
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        
        return combined
    
    def embed_sequence(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Embed sequence using values and temporal marks."""
        return self.forward(values=x, temporal_marks=x_mark)
    
    def get_embedding_type(self) -> str:
        return "hybrid"
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'type': 'hybrid',
            'strategy': self.combination_strategy,
            'components': [
                self.temporal_embedding.get_capabilities(),
                self.value_embedding.get_capabilities(),
                self.covariate_embedding.get_capabilities()
            ]
        }