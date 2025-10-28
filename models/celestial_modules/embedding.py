# models/celestial_modules/embedding.py

import torch
import torch.nn as nn
from .config import CelestialPGATConfig

# Import DataEmbedding directly to avoid circular imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from layers.modular.aggregation.phase_aware_celestial_processor import PhaseAwareCelestialProcessor
from layers.modular.embedding.calendar_aware_embedding import CalendarEffectsEncoder

# Temporary DataEmbedding implementation to avoid circular imports
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 25
        weekday_size = 8; day_size = 32; month_size = 13

        Embed = nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
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

import math

class EmbeddingModule(nn.Module):
    def __init__(self, config: CelestialPGATConfig):
        super().__init__()
        self.config = config

        if config.aggregate_waves_to_celestial:
            self.phase_aware_processor = PhaseAwareCelestialProcessor(
                num_input_waves=config.num_input_waves,
                celestial_dim=config.celestial_dim,
                waves_per_body=9,
                num_heads=config.n_heads
            )
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
            embedding_input_dim = config.enc_in
        
        self.enc_embedding = DataEmbedding(embedding_input_dim, config.d_model, config.embed, config.freq, config.dropout)
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
        # 1. Phase-Aware Wave Processing
        celestial_features, phase_based_adj = None, None
        if self.config.aggregate_waves_to_celestial:
            celestial_features, adjacency_matrix, _ = self.phase_aware_processor(x_enc)
            phase_based_adj = adjacency_matrix.unsqueeze(1).expand(-1, self.config.seq_len, -1, -1)
            x_enc_processed = self.celestial_projection(celestial_features)
        else:
            x_enc_processed = x_enc

        # 2. Encoder Embedding
        enc_out = self.enc_embedding(x_enc_processed, x_mark_enc)
        if self.config.use_calendar_effects and self.calendar_effects_encoder is not None:
            date_info = x_mark_enc[:, :, 0]
            calendar_embeddings = self.calendar_effects_encoder(date_info)
            combined = torch.cat([enc_out, calendar_embeddings], dim=-1)
            enc_out = self.calendar_fusion(combined)

        # 3. Decoder Embedding
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        if self.config.use_calendar_effects and self.calendar_effects_encoder is not None:
            dec_date_info = x_mark_dec[:, :, 0]
            dec_calendar_embeddings = self.calendar_effects_encoder(dec_date_info)
            dec_combined = torch.cat([dec_out, dec_calendar_embeddings], dim=-1)
            dec_out = self.calendar_fusion(dec_combined)
            
        return enc_out, dec_out, celestial_features, phase_based_adj, x_enc_processed