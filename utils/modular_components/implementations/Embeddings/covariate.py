import torch
import torch.nn as nn
from .base import BaseEmbedding
from ..config_schemas import EmbeddingConfig

class CovariateEmbedding(BaseEmbedding):
    """
    Covariate embedding for external features
    Handles both categorical and numerical covariates with appropriate embedding strategies.
    """
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.categorical_features = getattr(config, 'categorical_features', {})
        self.numerical_features = getattr(config, 'numerical_features', 0)
        self.categorical_embeddings = nn.ModuleDict()
        for feature_name, vocab_size in self.categorical_features.items():
            self.categorical_embeddings[feature_name] = nn.Embedding(vocab_size, self.d_model // max(len(self.categorical_features), 1))
        if self.numerical_features > 0:
            self.numerical_projection = nn.Linear(self.numerical_features, self.d_model)
        total_cat_dim = sum(self.d_model // max(len(self.categorical_features), 1) for _ in self.categorical_features)
        total_input_dim = total_cat_dim + (self.d_model if self.numerical_features > 0 else 0)
        self.combination = nn.Linear(total_input_dim, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, categorical_data=None, numerical_data=None):
        cat_embs = []
        if categorical_data is not None:
            for feature_name, values in categorical_data.items():
                if feature_name in self.categorical_embeddings:
                    cat_embs.append(self.categorical_embeddings[feature_name](values.long()))
        if cat_embs:
            cat_emb = torch.cat(cat_embs, dim=-1)
        else:
            cat_emb = None
        if self.numerical_features > 0 and numerical_data is not None:
            num_emb = self.numerical_projection(numerical_data)
        else:
            num_emb = None
        if cat_emb is not None and num_emb is not None:
            combined = torch.cat([cat_emb, num_emb], dim=-1)
        elif cat_emb is not None:
            combined = cat_emb
        elif num_emb is not None:
            combined = num_emb
        else:
            combined = torch.zeros(numerical_data.shape[0], numerical_data.shape[1], self.d_model, device=numerical_data.device)
        out = self.combination(combined)
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out
    def get_output_dim(self):
        return self.d_model
    def get_embedding_type(self):
        return "covariate"
