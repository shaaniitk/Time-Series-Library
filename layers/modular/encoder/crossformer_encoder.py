from .abstract_encoder import BaseEncoder
from models.Crossformer import Model as CrossformerModel
import torch
import torch.nn as nn
from typing import Tuple, Optional
from utils.logger import logger

class CrossformerEncoder(BaseEncoder):
    """
    Crossformer encoder wrapper for the modular encoder system.
    
    This encoder wraps the Crossformer model's encoder functionality to make it
    compatible with the modular encoder interface. It extracts and uses only the
    encoder part of the Crossformer architecture.
    
    Args:
        configs: Configuration object containing model parameters
    """
    
    def __init__(self, configs):
        super(CrossformerEncoder, self).__init__()
        logger.info("Initializing CrossformerEncoder wrapper")
        
        # Store configuration
        self.configs = configs
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        
        # Initialize Crossformer components needed for encoding
        from math import ceil
        from layers.Embed import PatchEmbedding
        from layers.Crossformer_EncDec import scale_block, Encoder
        
        # Segment and window parameters (from Crossformer)
        self.seg_len = 12
        self.win_size = 2
        
        # Padding operations
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        
        # Embedding layers
        self.enc_value_embedding = PatchEmbedding(
            configs.d_model, self.seg_len, self.seg_len, 
            self.pad_in_len - configs.seq_len, 0
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model)
        )
        self.pre_norm = nn.LayerNorm(configs.d_model)
        
        # Crossformer encoder
        self.encoder = Encoder([
            scale_block(
                configs, 
                1 if l == 0 else self.win_size, 
                configs.d_model, 
                configs.n_heads, 
                configs.d_ff,
                1, 
                configs.dropout,
                self.in_seg_num if l == 0 else ceil(self.in_seg_num / self.win_size ** l), 
                configs.factor
            ) for l in range(configs.e_layers)
        ])
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the Crossformer encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, enc_in)
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            
        Returns:
            tuple: A tuple containing:
                   - torch.Tensor: The encoder output
                   - torch.Tensor or None: The attention weights, if available
        """
        logger.debug(f"CrossformerEncoder forward - input shape: {x.shape}")
        
        # Apply Crossformer embedding process
        # Input: (batch_size, seq_len, enc_in)
        # Crossformer expects: (batch_size, enc_in, seq_len)
        x_permuted = x.permute(0, 2, 1)
        
        # Value embedding
        x_enc, n_vars = self.enc_value_embedding(x_permuted)
        
        # Rearrange to Crossformer format
        from einops import rearrange
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        
        # Add positional embedding
        x_enc += self.enc_pos_embedding
        
        # Pre-normalization
        x_enc = self.pre_norm(x_enc)
        
        # Apply Crossformer encoder
        enc_out, attns = self.encoder(x_enc)
        
        # Return the last layer output and attention weights
        # enc_out is a list of outputs from each layer
        final_output = enc_out[-1] if isinstance(enc_out, list) else enc_out
        
        logger.debug(f"CrossformerEncoder output shape: {final_output.shape}")
        
        return final_output, attns