# models/celestial_modules/decoder.py

import torch
import torch.nn as nn
from .config import CelestialPGATConfig

# Import components directly to avoid circular imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from layers.modular.decoder.target_autocorrelation_module import DualStreamDecoder
from layers.modular.graph.celestial_to_target_attention import CelestialToTargetAttention
from layers.modular.decoder.mdn_decoder import MDNDecoder

# Import DecoderLayer from the original model - we'll need to extract this
class DecoderLayer(nn.Module):
    """Temporary DecoderLayer implementation - should be extracted from original model"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, dec_input, enc_output):
        # Self-attention
        attn_output, _ = self.self_attention(dec_input, dec_input, dec_input)
        dec_input = self.norm1(dec_input + self.dropout(attn_output))
        
        # Cross-attention
        attn_output, _ = self.cross_attention(dec_input, enc_output, enc_output)
        dec_input = self.norm2(dec_input + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(dec_input)
        dec_input = self.norm3(dec_input + self.dropout(ff_output))
        
        return dec_input

class DecoderModule(nn.Module):
    def __init__(self, config: CelestialPGATConfig):
        super().__init__()
        self.config = config

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.dropout) for _ in range(config.d_layers)
        ])

        if config.use_target_autocorrelation:
            self.dual_stream_decoder = DualStreamDecoder(
                d_model=config.d_model, num_targets=config.c_out, num_heads=config.n_heads, dropout=config.dropout
            )
        else:
            self.dual_stream_decoder = None

        if config.use_celestial_target_attention:
            self.celestial_to_target_attention = CelestialToTargetAttention(
                num_celestial=config.num_celestial_bodies, num_targets=config.c_out, d_model=config.d_model,
                num_heads=config.n_heads, dropout=config.dropout, use_gated_fusion=config.celestial_target_use_gated_fusion
            )
        else:
            self.celestial_to_target_attention = None
            
        if config.enable_mdn_decoder:
            self.mdn_decoder = MDNDecoder(
                d_input=config.d_model, n_targets=config.c_out, n_components=config.mdn_components,
                sigma_min=config.mdn_sigma_min, use_softplus=config.mdn_use_softplus
            )
        else:
            self.mdn_decoder = None
            
        self.projection = nn.Linear(config.d_model, config.c_out)

    def forward(self, dec_out, graph_features, past_celestial_features, future_celestial_features):
        # 1. Standard Decoder Layers
        decoder_features = dec_out
        for layer in self.decoder_layers:
            decoder_features = layer(decoder_features, graph_features)

        # 2. Target Autocorrelation
        if self.config.use_target_autocorrelation and self.dual_stream_decoder is not None:
            decoder_features = self.dual_stream_decoder(decoder_features, graph_features)
            
        # 3. Celestial-to-Target Attention
        if self.config.use_celestial_target_attention and self.celestial_to_target_attention is not None:
            pred_start_idx = self.config.label_len
            decoder_pred_features = decoder_features[:, pred_start_idx:, :]
            decoder_target_features = decoder_pred_features.unsqueeze(2).expand(-1, -1, self.config.c_out, -1)
            
            # Prioritize known future celestial states
            celestial_feats = future_celestial_features if future_celestial_features is not None else past_celestial_features
            
            if celestial_feats is not None:
                # Ensure celestial_feats is 4D: [batch, seq_len, num_celestial, d_model]
                if celestial_feats.dim() == 3:
                    # Reshape from [batch, seq_len, celestial_feature_dim] to [batch, seq_len, num_celestial, celestial_dim]
                    batch_size, seq_len, celestial_feature_dim = celestial_feats.shape
                    celestial_dim = celestial_feature_dim // self.config.num_celestial_bodies
                    celestial_feats = celestial_feats.view(batch_size, seq_len, self.config.num_celestial_bodies, celestial_dim)
                    
                    # Project celestial features to d_model dimension if needed
                    if celestial_dim != self.config.d_model:
                        if not hasattr(self, 'celestial_projection'):
                            self.celestial_projection = nn.Linear(celestial_dim, self.config.d_model).to(celestial_feats.device)
                        # Reshape for projection: [batch*seq*num_celestial, celestial_dim]
                        batch_size, seq_len, num_celestial, celestial_dim = celestial_feats.shape
                        celestial_feats_flat = celestial_feats.view(-1, celestial_dim)
                        celestial_feats_projected = self.celestial_projection(celestial_feats_flat)
                        celestial_feats = celestial_feats_projected.view(batch_size, seq_len, num_celestial, self.config.d_model)
                
                enhanced_target_features, _ = self.celestial_to_target_attention(
                    target_features=decoder_target_features, celestial_features=celestial_feats
                )
                decoder_features_enhanced = enhanced_target_features.mean(dim=2)
                decoder_features = torch.cat([
                    decoder_features[:, :pred_start_idx, :],
                    decoder_features_enhanced
                ], dim=1)

        # 4. Final Prediction
        prediction_features = decoder_features[:, -self.config.pred_len:, :]
        if self.config.enable_mdn_decoder and self.mdn_decoder is not None:
            pi, mu, sigma = self.mdn_decoder(prediction_features)
            point_prediction = self.mdn_decoder.mean_prediction(pi, mu)
            return point_prediction, 0.0, (pi, mu, sigma) # aux_loss, mdn_components
        else:
            predictions = self.projection(prediction_features)
            return predictions, 0.0, None # aux_loss, mdn_components