import torch
import torch.nn as nn
from ..config_schemas import FFNConfig
from .base import BaseFeedForward

class StandardFFN(BaseFeedForward):
    """
    Standard feed-forward network used in vanilla Transformers
    Architecture: Linear -> Activation -> Dropout -> Linear -> Dropout
    """
    def __init__(self, config: FFNConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.dropout = nn.Dropout(config.dropout)
        self.activation = getattr(nn, config.activation.capitalize(), nn.ReLU)()
        self.linear1 = nn.Linear(self.d_model, self.d_ff, bias=config.use_bias)
        self.linear2 = nn.Linear(self.d_ff, self.d_model, bias=config.use_bias)
        self.layer_norm = nn.LayerNorm(self.d_model) if config.layer_norm else nn.Identity()
    def forward(self, x):
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out = self.layer_norm(out)
        return out
