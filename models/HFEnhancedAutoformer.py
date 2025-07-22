"""
Step 1: HF Enhanced Autoformer (Modularized)

Basic HF-based Enhanced Autoformer - refactored to use the modular component architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
import logging

from utils.modular_components.factories import create_backbone, create_embedding, create_output
from utils.modular_components.config_schemas import BackboneConfig, EmbeddingConfig, OutputConfig

logger = logging.getLogger(__name__)

class HFEnhancedAutoformer(nn.Module):
    """
    Modular HF-based Enhanced Autoformer
    
    This version is refactored to use the modular component architecture,
    making it more flexible and consistent with the rest of the library.
    """
    
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        
        logger.info("Initializing Modular HFEnhancedAutoformer")
        
        # Create Backbone Component
        backbone_config = BackboneConfig(
            backbone_type='robust_hf',
            model_name="amazon/chronos-t5-tiny",
            d_model=getattr(configs, 'd_model', 64),
            dropout=configs.dropout
        )
        self.backbone = create_backbone(backbone_config)
        self.d_model = self.backbone.get_output_dim()
        
        # Create Embedding Component
        embedding_config = EmbeddingConfig(
            embedding_type='hybrid',
            d_model=self.d_model,
            dropout=configs.dropout,
            custom_params={
                'embed_method': getattr(configs, 'embed', 'timeF'),
                'freq': getattr(configs, 'freq', 'h'),
                'use_value': True,
                'use_temporal': True
            }
        )
        self.embedding = create_embedding(embedding_config)
        
        # Create Output Head Component
        output_config = OutputConfig(
            output_type='forecasting',
            d_model=self.d_model,
            output_dim=configs.c_out,
            horizon=configs.pred_len,
            dropout=configs.dropout
        )
        self.output_head = create_output(output_config)
        
        logger.info(f"✅ Modular HFEnhancedAutoformer initialized successfully")
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Standard forward pass using modular components.
        """
        # Embedding
        enc_embed = self.embedding(values=x_enc, temporal_features=x_mark_enc)
        
        # Backbone processing
        if hasattr(self.backbone, 'supports_seq2seq') and self.backbone.supports_seq2seq():
            # Create decoder input
            dec_embed = self.embedding(values=x_dec, temporal_features=x_mark_dec)
            
            hidden_state = self.backbone(
                inputs_embeds=enc_embed,
                decoder_inputs_embeds=dec_embed
            )
        else:
            hidden_state = self.backbone(inputs_embeds=enc_embed)
        
        # Output head
        output = self.output_head(hidden_state)
        
        return output

# Export for testing
Model = HFEnhancedAutoformer