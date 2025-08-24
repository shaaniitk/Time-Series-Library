import logging
import torch
import torch.nn as nn
from typing import Optional, Dict
import torch
from ..base import BaseEmbedding
from ..core.configs import EmbeddingConfig

logger = logging.getLogger(__name__)

class CovariateEmbedding(BaseEmbedding):
    """
    Covariate embedding for external features
    
    Handles both categorical and numerical covariates with
    appropriate embedding strategies.
    """
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        
        # Covariate specifications
        self.categorical_features = getattr(config, 'categorical_features', {})
        self.numerical_features = getattr(config, 'numerical_features', 0)
        
        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleDict()
        for feature_name, vocab_size in self.categorical_features.items():
            self.categorical_embeddings[feature_name] = nn.Embedding(
                vocab_size, self.d_model // max(len(self.categorical_features), 1)
            )
        
        # Numerical feature projection
        if self.numerical_features > 0:
            self.numerical_projection = nn.Linear(self.numerical_features, self.d_model)
        
        # Combination layer
        total_cat_dim = sum(self.d_model // max(len(self.categorical_features), 1) 
                           for _ in self.categorical_features)
        total_input_dim = total_cat_dim + (self.d_model if self.numerical_features > 0 else 0)
        
        if total_input_dim > 0:
            self.combination_layer = nn.Linear(total_input_dim, self.d_model)
        else:
            self.combination_layer = nn.Identity()
        
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        logger.info(f"Covariate embedding initialized: categorical={len(self.categorical_features)}, numerical={self.numerical_features}")
    
    def forward(self, 
                categorical_data: Optional[Dict[str, torch.Tensor]] = None,
                numerical_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for covariate embedding
        
        Args:
            categorical_data: Dict of categorical features {name: [batch_size, seq_len]}
            numerical_data: Numerical features [batch_size, seq_len, num_features]
            
        Returns:
            Covariate embeddings [batch_size, seq_len, d_model]
        """
        embeddings_list = []
        
        # Process categorical features
        if categorical_data is not None:
            for feature_name, feature_values in categorical_data.items():
                if feature_name in self.categorical_embeddings:
                    emb = self.categorical_embeddings[feature_name](feature_values.long())
                    embeddings_list.append(emb)
        
        # Process numerical features
        if numerical_data is not None and self.numerical_features > 0:
            num_emb = self.numerical_projection(numerical_data)
            embeddings_list.append(num_emb)
        
        # Combine embeddings
        if embeddings_list:
            combined = torch.cat(embeddings_list, dim=-1)
            embeddings = self.combination_layer(combined)
        else:
            # Return zero embeddings if no covariates
            batch_size = 1
            seq_len = 1
            if categorical_data:
                for values in categorical_data.values():
                    batch_size, seq_len = values.shape[:2]
                    break
            elif numerical_data is not None:
                batch_size, seq_len = numerical_data.shape[:2]
            
            embeddings = torch.zeros(batch_size, seq_len, self.d_model,
                                   device=next(self.parameters()).device)
        
        # Apply normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def embed_sequence(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Embed covariates; treat x as numerical covariates."""
        return self.forward(categorical_data=None, numerical_data=x)
    
    def get_embedding_type(self) -> str:
        return "covariate"
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'type': 'covariate',
            'categorical_features': list(self.categorical_features.keys()),
            'numerical_features': self.numerical_features,
        }