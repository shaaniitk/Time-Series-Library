"""
Step 1: HF Enhanced Autoformer

Basic HF-based Enhanced Autoformer - the simplest replacement.
Drop-in replacement for EnhancedAutoformer using HF backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoConfig
import logging

logger = logging.getLogger(__name__)

class HFEnhancedAutoformer(nn.Module):
    """
    Basic HF-based Enhanced Autoformer
    
    This is the simplest replacement that focuses on:
    - Production stability using HF backbone
    - Clean, simple architecture
    - Drop-in compatibility with existing code
    """
    
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        
        logger.info("Initializing HFEnhancedAutoformer (Basic Enhanced Model)")
        logger.info(f"Config: enc_in={configs.enc_in}, c_out={configs.c_out}, seq_len={configs.seq_len}, pred_len={configs.pred_len}")
        
        # Try to use Chronos, fallback to standard transformer
        try:
            self.backbone = AutoModel.from_pretrained("amazon/chronos-t5-tiny")
            self.backbone_type = "chronos"
            logger.info("✅ Using Amazon Chronos T5 backbone")
        except Exception as e:
            logger.warning(f"Chronos not available ({e}), using fallback transformer")
            try:
                config = AutoConfig.from_pretrained("google/flan-t5-small")
                config.d_model = getattr(configs, 'd_model', 64)
                self.backbone = AutoModel.from_config(config)
                self.backbone_type = "t5"
                logger.info("✅ Using T5-small fallback backbone")
            except Exception as e2:
                logger.error(f"Both Chronos and T5 failed: {e2}")
                # Create minimal transformer config as last resort
                from transformers import T5Config, T5Model
                config = T5Config(
                    d_model=getattr(configs, 'd_model', 64),
                    num_heads=4,
                    num_layers=2,
                    d_ff=256,
                    vocab_size=1000
                )
                self.backbone = T5Model(config)
                self.backbone_type = "minimal"
                logger.info("✅ Using minimal T5 config as fallback")
        
        # Get backbone dimensions
        self.d_model = self.backbone.config.d_model
        logger.info(f"Backbone d_model: {self.d_model}")
        
        # Input projection layer (time series -> transformer space)
        self.input_projection = nn.Linear(configs.enc_in, self.d_model)
        
        # Output projection (transformer space -> prediction)
        self.output_projection = nn.Linear(self.d_model, configs.c_out)
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Positional encoding for time series
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max(configs.seq_len, configs.pred_len), self.d_model) * 0.02
        )
        
        logger.info(f"✅ HFEnhancedAutoformer initialized successfully")
        logger.info(f"   Input projection: {configs.enc_in} -> {self.d_model}")
        logger.info(f"   Output projection: {self.d_model} -> {configs.c_out}")
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Standard forward pass
        
        Args:
            x_enc: Encoder input (batch, seq_len, enc_in)
            x_mark_enc: Encoder time features (batch, seq_len, mark_size)
            x_dec: Decoder input (batch, pred_len, dec_in)  
            x_mark_dec: Decoder time features (batch, pred_len, mark_size)
            
        Returns:
            prediction: (batch, pred_len, c_out)
        """
        batch_size, seq_len, features = x_enc.shape
        
        # Project input to model dimension
        projected_input = self.input_projection(x_enc)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        pos_enc = self.positional_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
        projected_input = projected_input + pos_enc
        
        # Apply dropout
        projected_input = self.dropout(projected_input)
        
        # Handle different backbone types
        if self.backbone_type in ["chronos", "t5", "minimal"]:
            try:
                # Create decoder input for seq2seq models
                decoder_input = torch.zeros(
                    batch_size, self.configs.pred_len, self.d_model
                ).to(x_enc.device)
                
                # Add positional encoding to decoder input
                decoder_pos_enc = self.positional_encoding[:, :self.configs.pred_len, :].expand(batch_size, -1, -1)
                decoder_input = decoder_input + decoder_pos_enc
                
                # Forward through backbone
                outputs = self.backbone(
                    inputs_embeds=projected_input,
                    decoder_inputs_embeds=decoder_input
                )
                hidden_state = outputs.last_hidden_state  # (batch, pred_len, d_model)
                
            except Exception as e:
                logger.warning(f"Seq2seq forward failed ({e}), trying encoder-only")
                # Fallback: encoder-only mode
                if hasattr(self.backbone, 'encoder'):
                    outputs = self.backbone.encoder(inputs_embeds=projected_input)
                else:
                    outputs = self.backbone(inputs_embeds=projected_input)
                    
                encoder_hidden = outputs.last_hidden_state  # (batch, seq_len, d_model)
                
                # Pool encoder output to prediction length
                if encoder_hidden.shape[1] != self.configs.pred_len:
                    # Use adaptive pooling to get the right sequence length
                    encoder_hidden = encoder_hidden.transpose(1, 2)  # (batch, d_model, seq_len)
                    pooled = F.adaptive_avg_pool1d(encoder_hidden, self.configs.pred_len)
                    hidden_state = pooled.transpose(1, 2)  # (batch, pred_len, d_model)
                else:
                    hidden_state = encoder_hidden
        else:
            # Generic transformer handling
            outputs = self.backbone(inputs_embeds=projected_input)
            hidden_state = outputs.last_hidden_state
            
            # Ensure correct prediction length
            if hidden_state.shape[1] != self.configs.pred_len:
                hidden_state = hidden_state.transpose(1, 2)
                hidden_state = F.adaptive_avg_pool1d(hidden_state, self.configs.pred_len)
                hidden_state = hidden_state.transpose(1, 2)
        
        # Generate final prediction
        output = self.output_projection(hidden_state)  # (batch, pred_len, c_out)
        
        return output

# Export for testing
Model = HFEnhancedAutoformer
