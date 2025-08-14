# FILE: models/TwoTowerMoEModel.py
import torch
import torch.nn as nn
from utils.implementations.encoder.wrapped_encoders import EncoderProcessor
from utils.implementations.fusion.bi_attention_fusion import BiDirectionalFusionProcessor
from layers.Autoformer_EncDec import Decoder

class TwoTowerMoEModel(nn.Module):
    def __init__(self, target_encoder: EncoderProcessor, 
                 covariate_encoder: EncoderProcessor,
                 fusion_processor: BiDirectionalFusionProcessor,
                 decoder: Decoder,
                 d_model: int):
        super().__init__()
        self.target_encoder = target_encoder
        self.covariate_encoder = covariate_encoder
        self.fusion_processor = fusion_processor
        self.decoder = decoder
        
        # The fusion processor outputs 2 * d_model, which the decoder's 
        # cross-attention might not expect. This projection layer ensures compatibility.
        self.context_projection = nn.Linear(2 * d_model, d_model)

    def forward(self, target_series, covariate_series, 
                dec_input, target_mark=None, dec_mark=None):
        
        # Process targets and covariates through their respective towers
        target_repr = self.target_encoder(target_series)
        covariate_repr = self.covariate_encoder(covariate_series)
        
        # Fuse the representations
        # Note: The BiDirectionalFusionProcessor returns a tensor of shape [B, L, 2 * d_model]
        enriched_context = self.fusion_processor(target_repr, covariate_repr)
        
        # Project the enriched context back to the expected d_model dimension for the decoder
        projected_context = self.context_projection(enriched_context)
        
        # Decode using the enriched and projected context
        decoder_output = self.decoder(
            dec_input, 
            projected_context, 
            x_mask=None, 
            cross_mask=None
        )
        
        return decoder_output
