"""Simplified PGAT model for memory efficiency and stability."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class SimplifiedPGAT(nn.Module):
    """Simplified PGAT model focusing on memory efficiency and stability."""
    
    def __init__(self, config, mode='probabilistic'):
        super().__init__()
        self.config = config
        self.mode = mode
        self.d_model = getattr(config, 'd_model', 256)
        
        # Input dimensions
        self.enc_in = getattr(config, 'enc_in', 7)
        self.c_out = getattr(config, 'c_out', 1)
        self.seq_len = getattr(config, 'seq_len', 96)
        self.pred_len = getattr(config, 'pred_len', 24)
        
        print(f"SimplifiedPGAT initialized with: enc_in={self.enc_in}, c_out={self.c_out}, d_model={self.d_model}")
        
        # Simple embedding layer
        self.embedding = nn.Linear(self.enc_in, self.d_model)
        
        # Simple transformer layers
        self.num_layers = getattr(config, 'e_layers', 2)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=getattr(config, 'n_heads', 4),
                dim_feedforward=getattr(config, 'd_ff', 512),
                dropout=getattr(config, 'dropout', 0.1),
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.c_out)
        
        # Normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
        
    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark) -> torch.Tensor:
        """Simplified forward pass."""
        batch_size = batch_x.shape[0]
        
        # Use only the encoder input for simplicity
        # batch_x shape: [batch_size, seq_len, features]
        x = batch_x
        
        # Embedding
        embedded = self.embedding(x)
        embedded = self.layer_norm(embedded)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            embedded = layer(embedded)
        
        # Output projection - take the last pred_len timesteps
        # For forecasting, we want to predict the next pred_len steps
        output = self.output_projection(embedded)
        
        # Return only the prediction length portion
        if output.shape[1] > self.pred_len:
            output = output[:, -self.pred_len:, :]
        
        # Ensure output has the correct feature dimension
        # The framework expects [batch_size, pred_len, c_out]
        return output
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            'model_parameters': sum(p.numel() for p in self.parameters()),
            'model_type': 'SimplifiedPGAT',
        }