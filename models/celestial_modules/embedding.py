# models/celestial_modules/embedding.py

import math
import torch
import torch.nn as nn
from .config import CelestialPGATConfig

# Import components directly to avoid circular imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from layers.modular.aggregation.phase_aware_celestial_processor import PhaseAwareCelestialProcessor
from layers.modular.embedding.calendar_aware_embedding import CalendarEffectsEncoder
from utils.celestial_wave_aggregator import CelestialWaveAggregator, CelestialDataProcessor

# Enhanced DataEmbedding implementation with proper validation
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # Validate input dimensions
        batch_size, seq_len, input_dim = x.shape
        if input_dim != self.c_in:
            raise ValueError(
                f"DataEmbedding input_dim ({input_dim}) does not match expected c_in ({self.c_in})."
                f" Ensure encoder/embedding configuration is consistent."
            )
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # Enforce fixed input dimension for stability
        batch_size, seq_len, input_dim = x.shape
        if input_dim != self.c_in:
            raise ValueError(
                f"TokenEmbedding input_dim ({input_dim}) does not match expected c_in ({self.c_in})."
                f" Ensure encoder/embedding configuration is consistent."
            )
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe.requires_grad = False
        
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # Use separate div terms for even and odd indices to handle odd d_model robustly
        div_term_even = (torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)).exp()
        div_term_odd = (torch.arange(1, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)).exp()

        # Shapes:
        # - pe[:, 0::2] -> [max_len, ceil(d_model/2)]
        # - pe[:, 1::2] -> [max_len, floor(d_model/2)]
        # Ensure broadcasted multiplications match target slice widths
        pe[:, 0::2] = torch.sin(position * div_term_even)
        if div_term_odd.numel() > 0:
            pe[:, 1::2] = torch.cos(position * div_term_odd)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 25
        weekday_size = 8
        day_size = 32
        month_size = 13

        Embed = nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        # Handle variable number of time features gracefully
        num_features = x.size(-1)
        
        # Initialize embeddings to zero
        minute_x = 0.
        hour_x = 0.
        weekday_x = 0.
        day_x = 0.
        month_x = 0.
        
        # Apply embeddings based on available features
        # Convert continuous temporal features to discrete indices
        if num_features > 0:
            month_indices = torch.clamp((x[:, :, 0] * 12).long(), 0, 11)
            month_x = self.month_embed(month_indices)
        if num_features > 1:
            day_indices = torch.clamp((x[:, :, 1] * 31).long(), 0, 30)
            day_x = self.day_embed(day_indices)
        if num_features > 2:
            weekday_indices = torch.clamp((x[:, :, 2] * 7).long(), 0, 6)
            weekday_x = self.weekday_embed(weekday_indices)
        if num_features > 3:
            hour_indices = torch.clamp((x[:, :, 3] * 24).long(), 0, 23)
            hour_x = self.hour_embed(hour_indices)
        if num_features > 4 and hasattr(self, 'minute_embed'):
            minute_indices = torch.clamp((x[:, :, 4] * 60).long(), 0, 59)
            minute_x = self.minute_embed(minute_indices)
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map.get(freq, 4)  # Default to 4 if freq not found
        self.embed = nn.Linear(d_inp, d_model, bias=False)
    
    def forward(self, x):
        # Handle case where input has more features than expected
        if x.size(-1) > self.embed.in_features:
            x = x[..., :self.embed.in_features]
        return self.embed(x)

class EmbeddingModule(nn.Module):
    def __init__(self, config: CelestialPGATConfig):
        super().__init__()
        self.config = config

        # Enhanced Phase-Aware Wave Processing System
        if config.aggregate_waves_to_celestial:
            self.phase_aware_processor = PhaseAwareCelestialProcessor(
                num_input_waves=config.num_input_waves,
                celestial_dim=config.celestial_dim,
                waves_per_body=9,
                num_heads=config.n_heads
            )
            
            # Update num_input_waves with auto-detected value from processor
            config.num_input_waves = self.phase_aware_processor.num_input_waves
            
            # Keep the old processor for target extraction
            self.wave_aggregator = CelestialWaveAggregator(
                num_input_waves=config.num_input_waves,
                num_celestial_bodies=config.num_celestial_bodies
            )
            self.data_processor = CelestialDataProcessor(
                self.wave_aggregator,
                target_indices=config.target_wave_indices
            )
            
            # New projection layer to fix information bottleneck
            self.rich_feature_to_celestial = nn.Linear(config.celestial_feature_dim, config.num_celestial_bodies)
            
            self.celestial_projection = nn.Sequential(
                nn.Linear(config.celestial_feature_dim, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )
            embedding_input_dim = config.d_model
        else:
            self.phase_aware_processor = None
            self.celestial_projection = None
            self.wave_aggregator = None
            self.data_processor = None
            self.rich_feature_to_celestial = None
            embedding_input_dim = config.enc_in
        
        # Store expected embedding input dimension
        config.expected_embedding_input_dim = embedding_input_dim
        
        self.enc_embedding = DataEmbedding(embedding_input_dim, config.d_model, config.embed, config.freq, config.dropout)
        # Decoder always uses dec_in dimensions (targets don't go through celestial aggregation)
        self.dec_embedding = DataEmbedding(config.dec_in, config.d_model, config.embed, config.freq, config.dropout)

        if config.use_calendar_effects:
            self.calendar_effects_encoder = CalendarEffectsEncoder(config.calendar_embedding_dim)
            self.calendar_fusion = nn.Sequential(
                nn.Linear(config.d_model + config.calendar_embedding_dim, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )
        else:
            self.calendar_effects_encoder = None
            self.calendar_fusion = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        batch_size, seq_len, enc_in = x_enc.shape
        
        # 1. Enhanced Phase-Aware Wave Processing
        wave_metadata = {}
        phase_based_adj = None
        target_shape = (batch_size, seq_len, len(self.config.target_wave_indices))
        
        if self.config.aggregate_waves_to_celestial and enc_in >= self.config.num_input_waves:
            # Use phase-aware processor for rich celestial representations
            celestial_features, adjacency_matrix, phase_metadata = self.phase_aware_processor(x_enc)
            # celestial_features: [batch, seq_len, 13 * 32] Rich multi-dimensional representations
            # adjacency_matrix: [batch, 13, 13] Phase-difference based edges
            phase_based_adj = adjacency_matrix.unsqueeze(1).expand(-1, seq_len, -1, -1)
            
            # Process with old data processor for target extraction (if diagnostics needed)
            target_metadata = {}
            target_waves = None
            if self.config.collect_diagnostics and self.data_processor is not None:
                try:
                    target_waves, target_metadata = self.data_processor.process_batch(x_enc)
                except Exception as e:
                    # Handle dimension mismatch gracefully
                    target_metadata = {'diagnostics_disabled': True, 'error': str(e)}
                    target_waves = None
            
            # Apply celestial projection to match d_model
            x_enc_processed = self.celestial_projection(celestial_features)
            
            # Store comprehensive metadata
            if self.config.collect_diagnostics:
                wave_metadata = {
                    'original_targets': target_waves,
                    'phase_metadata': phase_metadata,
                    'target_metadata': target_metadata,
                    'adjacency_matrix_stats': {
                        'mean': adjacency_matrix.mean().item(),
                        'std': adjacency_matrix.std().item(),
                    },
                    'celestial_features_shape': celestial_features.shape,
                }
        else:
            x_enc_processed = x_enc
            celestial_features = None
            wave_metadata['original_targets'] = None

        # 2. Encoder Embedding with validation
        try:
            enc_out = self.enc_embedding(x_enc_processed, x_mark_enc)
            
            # Apply Calendar Effects Enhancement
            if self.config.use_calendar_effects and self.calendar_effects_encoder is not None:
                date_info = x_mark_enc[:, :, 0]
                calendar_embeddings = self.calendar_effects_encoder(date_info)
                combined_features = torch.cat([enc_out, calendar_embeddings], dim=-1)
                enc_out = self.calendar_fusion(combined_features)
                
        except Exception as exc:
            raise ValueError(
                f"Encoder embedding failed | input_shape={x_enc_processed.shape} "
                f"expected_dim={self.config.expected_embedding_input_dim} "
                f"embedding_c_in={self.enc_embedding.c_in}"
            ) from exc

        # 3. Decoder Embedding
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # Apply Calendar Effects to Decoder
        if self.config.use_calendar_effects and self.calendar_effects_encoder is not None:
            dec_date_info = x_mark_dec[:, :, 0]
            dec_calendar_embeddings = self.calendar_effects_encoder(dec_date_info)
            dec_combined_features = torch.cat([dec_out, dec_calendar_embeddings], dim=-1)
            dec_out = self.calendar_fusion(dec_combined_features)
            
        return enc_out, dec_out, celestial_features, phase_based_adj, x_enc_processed, wave_metadata