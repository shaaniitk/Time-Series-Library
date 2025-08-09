"""
Concrete Embedding Implementations

This module provides concrete implementations of the BaseEmbedding interface
for different embedding strategies in time series forecasting.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
import math

from ..base_interfaces import BaseEmbedding
from ..config_schemas import EmbeddingConfig

logger = logging.getLogger(__name__)


class TemporalEmbedding(BaseEmbedding):
    """
    Temporal embedding with multiple time feature support
    
    Combines positional encoding with temporal features like
    day of week, hour of day, month, etc.
    """
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        self.max_len = getattr(config, 'max_len', 5000)
        self.temp_feature_dim = getattr(config, 'temp_feature_dim', 4)  # day, hour, month, weekday
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding()
        
        # Temporal feature embeddings
        self.temporal_embeddings = nn.ModuleDict({
            'hour': nn.Embedding(24, self.d_model // 4),
            'day': nn.Embedding(32, self.d_model // 4),  # day of month
            'weekday': nn.Embedding(7, self.d_model // 4),
            'month': nn.Embedding(13, self.d_model // 4)  # 1-12
        })
        
        # Projection layer
        self.projection = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        logger.info(f"Temporal embedding initialized with d_model={self.d_model}")
    
    def _create_positional_encoding(self) -> nn.Parameter:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, 
                input_embeddings: torch.Tensor,
                temporal_features: Optional[Dict[str, torch.Tensor]] = None,
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for temporal embedding
        
        Args:
            input_embeddings: Base embeddings [batch_size, seq_len, d_model]
            temporal_features: Dict of temporal features {feature_name: [batch_size, seq_len]}
            positions: Position indices [batch_size, seq_len]
            
        Returns:
            Enhanced embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = input_embeddings.size()
        
        # Start with input embeddings
        embeddings = input_embeddings
        
        # Add positional encoding
        if positions is not None:
            pos_embeddings = self.positional_encoding[:, positions.long(), :]
        else:
            pos_embeddings = self.positional_encoding[:, :seq_len, :]
        
        embeddings = embeddings + pos_embeddings
        
        # Add temporal features if provided
        if temporal_features is not None:
            temporal_emb = self._process_temporal_features(temporal_features)
            embeddings = embeddings + temporal_emb
        
        # Apply projection and normalization
        embeddings = self.projection(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def _process_temporal_features(self, temporal_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process temporal features into embeddings"""
        temporal_embs = []
        first_feat: Optional[torch.Tensor] = None
        
        for feature_name, feature_values in temporal_features.items():
            if first_feat is None:
                first_feat = feature_values
            if feature_name in self.temporal_embeddings:
                emb = self.temporal_embeddings[feature_name](feature_values.long())
                temporal_embs.append(emb)
        
        if temporal_embs:
            # Concatenate and project to d_model
            combined = torch.cat(temporal_embs, dim=-1)
            # Pad or truncate to d_model
            if combined.size(-1) < self.d_model:
                padding = torch.zeros(*combined.shape[:-1], 
                                    self.d_model - combined.size(-1),
                                    device=combined.device)
                combined = torch.cat([combined, padding], dim=-1)
            elif combined.size(-1) > self.d_model:
                combined = combined[..., :self.d_model]
            
            return combined
        
        # If no recognized features, return zeros with correct shape
        if first_feat is not None:
            b, l = first_feat.shape[:2]
            return torch.zeros(b, l, self.d_model, device=first_feat.device)
        return torch.zeros(1, 1, self.d_model)
    
    def embed_sequence(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Alias to forward using x as base embeddings; x_mark ignored."""
        return self.forward(input_embeddings=x, temporal_features=None, positions=None)
    
    def get_embedding_type(self) -> str:
        return "temporal"
    
    def get_output_dim(self) -> int:
        return self.d_model
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            'type': 'temporal',
            'max_sequence_length': self.max_len,
            'temporal_features': list(self.temporal_embeddings.keys()),
        }


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


class HybridEmbedding(BaseEmbedding):
    """
    Hybrid embedding that combines multiple embedding strategies
    
    Allows combining temporal, value, and covariate embeddings
    for comprehensive feature representation.
    """
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        
        # Component embeddings
        self.use_temporal = getattr(config, 'use_temporal', True)
        self.use_value = getattr(config, 'use_value', True)
        self.use_covariate = getattr(config, 'use_covariate', False)
        
        self.embeddings = nn.ModuleDict()
        
        if self.use_temporal:
            temporal_config = EmbeddingConfig(
                d_model=self.d_model,
                dropout=config.dropout,
                **getattr(config, 'temporal_config', {})
            )
            self.embeddings['temporal'] = TemporalEmbedding(temporal_config)
        
        if self.use_value:
            value_config = EmbeddingConfig(
                d_model=self.d_model,
                dropout=config.dropout,
                **getattr(config, 'value_config', {})
            )
            self.embeddings['value'] = ValueEmbedding(value_config)
        
        if self.use_covariate:
            covariate_config = EmbeddingConfig(
                d_model=self.d_model,
                dropout=config.dropout,
                **getattr(config, 'covariate_config', {})
            )
            self.embeddings['covariate'] = CovariateEmbedding(covariate_config)
        
        # Combination strategy
        self.combination_strategy = getattr(config, 'combination_strategy', 'add')  # 'add', 'concat', 'weighted'
        
        if self.combination_strategy == 'weighted':
            num_components = sum([self.use_temporal, self.use_value, self.use_covariate])
            self.combination_weights = nn.Parameter(torch.ones(num_components) / num_components)
        
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        logger.info(f"Hybrid embedding initialized: temporal={self.use_temporal}, value={self.use_value}, covariate={self.use_covariate}")
    
    def forward(self, 
                values: Optional[torch.Tensor] = None,
                temporal_features: Optional[Dict[str, torch.Tensor]] = None,
                categorical_data: Optional[Dict[str, torch.Tensor]] = None,
                numerical_data: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass for hybrid embedding
        
        Args:
            values: Input values for value embedding
            temporal_features: Temporal features for temporal embedding
            categorical_data: Categorical covariates
            numerical_data: Numerical covariates
            **kwargs: Additional arguments
            
        Returns:
            Combined embeddings [batch_size, seq_len, d_model]
        """
        component_embeddings = []
        
        # Get value embeddings as base
        if self.use_value and values is not None:
            value_emb = self.embeddings['value'](values)
            component_embeddings.append(value_emb)
            base_shape = value_emb.shape
        else:
            # Create base embedding shape from other inputs
            if temporal_features:
                first_temp_feature = next(iter(temporal_features.values()))
                base_shape = (first_temp_feature.shape[0], first_temp_feature.shape[1], self.d_model)
            elif categorical_data:
                first_cat_feature = next(iter(categorical_data.values()))
                base_shape = (first_cat_feature.shape[0], first_cat_feature.shape[1], self.d_model)
            elif numerical_data is not None:
                base_shape = (numerical_data.shape[0], numerical_data.shape[1], self.d_model)
            else:
                raise ValueError("At least one input type must be provided")
            
            # Create zero base embedding
            device = next(self.parameters()).device
            base_emb = torch.zeros(*base_shape, device=device)
            component_embeddings.append(base_emb)
        
        # Get temporal embeddings
        if self.use_temporal and temporal_features is not None:
            # Use base embedding as input for temporal enhancement
            temporal_emb = self.embeddings['temporal'](
                component_embeddings[0], 
                temporal_features=temporal_features,
                **kwargs
            )
            component_embeddings.append(temporal_emb)
        
        # Get covariate embeddings
        if self.use_covariate and (categorical_data is not None or numerical_data is not None):
            covariate_emb = self.embeddings['covariate'](
                categorical_data=categorical_data,
                numerical_data=numerical_data
            )
            component_embeddings.append(covariate_emb)
        
        # Combine embeddings
        if len(component_embeddings) == 1:
            combined = component_embeddings[0]
        elif self.combination_strategy == 'add':
            combined = sum(component_embeddings)
        elif self.combination_strategy == 'weighted':
            weights = torch.softmax(self.combination_weights, dim=0)
            combined = sum(w * emb for w, emb in zip(weights, component_embeddings))
        else:  # Default to add
            combined = sum(component_embeddings)
        
        # Apply final normalization
        combined = self.layer_norm(combined)
        
        return combined
    
    def embed_sequence(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Embed sequence; treat x as base values."""
        return self.forward(values=x)
    
    def get_embedding_type(self) -> str:
        return "hybrid"
    
    def get_output_dim(self) -> int:
        """Get output embedding dimension"""
        return self.d_model
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get embedding capabilities"""
        capabilities = {
            'type': 'hybrid',
            'combination_strategy': self.combination_strategy,
            'components': []
        }
        
        for name, embedding in self.embeddings.items():
            capabilities['components'].append({
                'name': name,
                'capabilities': embedding.get_capabilities()
            })
        
        return capabilities
