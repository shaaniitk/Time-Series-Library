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

class EmbeddingError(Exception):
    """Custom exception for embedding-related errors"""
    pass

class DimensionMismatchError(EmbeddingError):
    """Raised when input dimensions don't match expected dimensions"""
    pass

class CelestialProcessingError(EmbeddingError):
    """Raised when celestial processing fails"""
    pass

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
        if freq not in freq_map:
            raise ValueError(f"Unsupported frequency '{freq}'. Supported frequencies: {list(freq_map.keys())}")
        
        d_inp = freq_map[freq]
        self.expected_features = d_inp
        self.embed = nn.Linear(d_inp, d_model, bias=False)
    
    def forward(self, x):
        # STRICT validation - no silent truncation
        if x.size(-1) != self.expected_features:
            raise DimensionMismatchError(
                f"TimeFeatureEmbedding expects {self.expected_features} time features, "
                f"but got {x.size(-1)}. Input shape: {x.shape}. "
                f"Ensure time feature preprocessing matches frequency configuration."
            )
        return self.embed(x)

class EmbeddingModule(nn.Module):
    def __init__(self, config: CelestialPGATConfig):
        super().__init__()
        self.config = config

        # Enhanced Phase-Aware Wave Processing System
        # FIX ISSUE #5: Add explicit phase-aware processing flag check
        if config.aggregate_waves_to_celestial:
            # Check if phase-aware processing is requested (default: True)
            use_phase_aware = getattr(config, 'use_phase_aware_processing', True)
            
            if use_phase_aware:
                self.phase_aware_processor = PhaseAwareCelestialProcessor(
                    num_input_waves=config.num_input_waves,
                    celestial_dim=config.celestial_dim,
                    waves_per_body=9,
                    num_heads=config.n_heads
                )
                
                # Update num_input_waves with auto-detected value from processor
                config.num_input_waves = self.phase_aware_processor.num_input_waves
            else:
                # Simple aggregation without phase-aware processing
                self.phase_aware_processor = None
            
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
        
        # Decoder embedding configuration - STRICT dimension handling
        if config.aggregate_waves_to_celestial:
            # When using celestial processing, decoder input will be projected via celestial_projection
            decoder_embedding_input_dim = config.d_model
        else:
            # When not using celestial processing, we need a raw projection layer
            decoder_embedding_input_dim = config.d_model
            self.raw_dec_projection = nn.Linear(config.dec_in, config.d_model)
            
        self.dec_embedding = DataEmbedding(decoder_embedding_input_dim, config.d_model, config.embed, config.freq, config.dropout)

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
        # DTYPE FIX: Ensure all inputs are float32
        x_enc = x_enc.float()
        x_mark_enc = x_mark_enc.float()
        x_dec = x_dec.float()
        x_mark_dec = x_mark_dec.float()
        
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

        # 3. Decoder Embedding - STRICT: No fallbacks, proper error handling
        batch_size_dec, seq_len_dec, dec_in = x_dec.shape
        
        if self.config.aggregate_waves_to_celestial:
            if dec_in < self.config.num_input_waves:
                raise DimensionMismatchError(
                    f"Decoder input has {dec_in} features but celestial processing requires "
                    f"{self.config.num_input_waves} input waves. "
                    f"Either disable celestial processing or ensure decoder input has sufficient features."
                )
            
            # Apply celestial processing to decoder input for rich feature representation
            try:
                celestial_features_dec, _, _ = self.phase_aware_processor(x_dec)
                x_dec_processed = self.celestial_projection(celestial_features_dec)
                
                if getattr(self.config, 'debug_mode', False):
                    print(f"Decoder: Applied celestial processing {x_dec.shape} → {celestial_features_dec.shape} → {x_dec_processed.shape}")
                    
            except Exception as e:
                raise CelestialProcessingError(
                    f"Decoder celestial processing failed: {e}. "
                    f"Input shape: {x_dec.shape}, Expected waves: {self.config.num_input_waves}. "
                    f"Check celestial processor configuration and input data consistency."
                ) from e
        else:
            # When not using celestial processing, we need a proper projection layer
            # This should be configured at initialization, not created on-the-fly
            if not hasattr(self, 'raw_dec_projection'):
                raise EmbeddingError(
                    f"Decoder projection layer not initialized. "
                    f"When celestial processing is disabled, decoder input projection must be configured. "
                    f"Input shape: {x_dec.shape}, Expected output: d_model={self.config.d_model}"
                )
            
            x_dec_processed = self.raw_dec_projection(x_dec)
            
        dec_out = self.dec_embedding(x_dec_processed, x_mark_dec)
        
        # Apply Calendar Effects to Decoder
        if self.config.use_calendar_effects and self.calendar_effects_encoder is not None:
            dec_date_info = x_mark_dec[:, :, 0]
            dec_calendar_embeddings = self.calendar_effects_encoder(dec_date_info)
            dec_combined_features = torch.cat([dec_out, dec_calendar_embeddings], dim=-1)
            dec_out = self.calendar_fusion(dec_combined_features)
            
        return enc_out, dec_out, celestial_features, phase_based_adj, x_enc_processed, wave_metadata