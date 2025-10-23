"""
Calendar-Aware Temporal Embedding

This module creates sophisticated temporal embeddings that capture calendar effects
important for financial markets:
1. End-of-month effects
2. End-of-week effects  
3. Last weekday effects
4. Holiday proximity effects
5. Quarter-end effects
6. Year-end effects
7. Day-of-week anomalies
8. Month-of-year seasonality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import math


class CalendarEffectsEncoder(nn.Module):
    """
    Encodes various calendar effects that are known to impact financial markets
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Day of week effects (Monday effect, Friday effect, etc.)
        self.dow_embedding = nn.Embedding(7, d_model // 8)  # 0=Monday, 6=Sunday
        
        # Month of year effects (January effect, December effect, etc.)
        self.month_embedding = nn.Embedding(12, d_model // 8)  # 0=January, 11=December
        
        # Quarter effects
        self.quarter_embedding = nn.Embedding(4, d_model // 16)  # 0=Q1, 3=Q4
        
        # End-of-period effects
        self.end_of_month_encoder = nn.Linear(1, d_model // 16)
        self.end_of_quarter_encoder = nn.Linear(1, d_model // 16)
        self.end_of_year_encoder = nn.Linear(1, d_model // 16)
        
        # Weekday vs weekend
        self.is_weekday_encoder = nn.Linear(1, d_model // 16)
        
        # Days until/since important dates
        self.days_to_month_end_encoder = nn.Linear(1, d_model // 16)
        self.days_to_quarter_end_encoder = nn.Linear(1, d_model // 16)
        
        # Holiday proximity (simplified - can be enhanced with actual holiday calendars)
        self.holiday_proximity_encoder = nn.Linear(1, d_model // 16)
        
        # Combine all calendar features
        total_calendar_dim = (
            d_model // 8 +  # dow
            d_model // 8 +  # month  
            d_model // 16 + # quarter
            d_model // 16 + # end_of_month
            d_model // 16 + # end_of_quarter
            d_model // 16 + # end_of_year
            d_model // 16 + # is_weekday
            d_model // 16 + # days_to_month_end
            d_model // 16 + # days_to_quarter_end
            d_model // 16   # holiday_proximity
        )
        
        self.calendar_projection = nn.Sequential(
            nn.Linear(total_calendar_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def extract_calendar_features(self, dates: torch.Tensor) -> dict:
        """
        Extract calendar features from date tensor
        
        Args:
            dates: [batch, seq_len] - Unix timestamps or date indices
            
        Returns:
            Dictionary of calendar features
        """
        # This is a simplified version - in practice, you'd convert timestamps to actual dates
        # For now, we'll simulate calendar features
        batch_size, seq_len = dates.shape
        device = dates.device
        
        # Simulate day of week (0-6, Monday=0)
        dow = (dates % 7).long()
        
        # Simulate month (0-11)
        month = ((dates // 30) % 12).long()
        
        # Simulate quarter (0-3)
        quarter = (month // 3).long()
        
        # End of month indicator (last 3 days of month)
        day_of_month = (dates % 30) + 1
        end_of_month = (day_of_month >= 28).float().unsqueeze(-1)
        
        # End of quarter indicator (last week of quarter)
        days_in_quarter = (dates % 90) + 1
        end_of_quarter = (days_in_quarter >= 83).float().unsqueeze(-1)
        
        # End of year indicator (last month of year)
        end_of_year = (month >= 11).float().unsqueeze(-1)
        
        # Weekday indicator (Monday-Friday = 1, Weekend = 0)
        is_weekday = (dow < 5).float().unsqueeze(-1)
        
        # Days to month end (normalized)
        days_to_month_end = ((30 - day_of_month) / 30.0).unsqueeze(-1)
        
        # Days to quarter end (normalized)
        days_to_quarter_end = ((90 - days_in_quarter) / 90.0).unsqueeze(-1)
        
        # Holiday proximity (simplified - distance to major holidays)
        # This could be enhanced with actual holiday calendars
        holiday_proximity = torch.sin(dates * 2 * math.pi / 365.25).unsqueeze(-1)
        
        return {
            'dow': dow,
            'month': month,
            'quarter': quarter,
            'end_of_month': end_of_month,
            'end_of_quarter': end_of_quarter,
            'end_of_year': end_of_year,
            'is_weekday': is_weekday,
            'days_to_month_end': days_to_month_end,
            'days_to_quarter_end': days_to_quarter_end,
            'holiday_proximity': holiday_proximity
        }
    
    def forward(self, dates: torch.Tensor) -> torch.Tensor:
        """
        Create calendar-aware embeddings
        
        Args:
            dates: [batch, seq_len] - Date information
            
        Returns:
            Calendar embeddings [batch, seq_len, d_model]
        """
        calendar_features = self.extract_calendar_features(dates)
        
        # Create embeddings for categorical features
        dow_emb = self.dow_embedding(calendar_features['dow'])
        month_emb = self.month_embedding(calendar_features['month'])
        quarter_emb = self.quarter_embedding(calendar_features['quarter'])
        
        # Create embeddings for continuous features
        end_of_month_emb = self.end_of_month_encoder(calendar_features['end_of_month'])
        end_of_quarter_emb = self.end_of_quarter_encoder(calendar_features['end_of_quarter'])
        end_of_year_emb = self.end_of_year_encoder(calendar_features['end_of_year'])
        is_weekday_emb = self.is_weekday_encoder(calendar_features['is_weekday'])
        days_to_month_end_emb = self.days_to_month_end_encoder(calendar_features['days_to_month_end'])
        days_to_quarter_end_emb = self.days_to_quarter_end_encoder(calendar_features['days_to_quarter_end'])
        holiday_proximity_emb = self.holiday_proximity_encoder(calendar_features['holiday_proximity'])
        
        # Concatenate all calendar embeddings
        calendar_concat = torch.cat([
            dow_emb,
            month_emb,
            quarter_emb,
            end_of_month_emb,
            end_of_quarter_emb,
            end_of_year_emb,
            is_weekday_emb,
            days_to_month_end_emb,
            days_to_quarter_end_emb,
            holiday_proximity_emb
        ], dim=-1)
        
        # Project to d_model dimensions
        calendar_embedding = self.calendar_projection(calendar_concat)
        
        return calendar_embedding


class EnhancedTemporalEmbedding(nn.Module):
    """
    Enhanced temporal embedding that combines:
    1. Standard positional encoding
    2. Calendar effects encoding
    3. Time-of-day effects (if applicable)
    4. Seasonal patterns
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Standard positional encoding
        self.pos_encoding = self._create_positional_encoding(max_len, d_model // 2)
        
        # Calendar effects encoder
        self.calendar_encoder = CalendarEffectsEncoder(d_model // 2)
        
        # Combine positional and calendar embeddings
        self.temporal_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create standard sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def forward(
        self, 
        x: torch.Tensor, 
        dates: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create enhanced temporal embeddings
        
        Args:
            x: Input tensor [batch, seq_len, features]
            dates: Date information [batch, seq_len] (optional)
            
        Returns:
            Enhanced temporal embeddings [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Standard positional encoding
        pos_emb = self.pos_encoding[:, :seq_len, :].to(device)
        pos_emb = pos_emb.expand(batch_size, -1, -1)
        
        # Calendar effects encoding
        if dates is not None:
            calendar_emb = self.calendar_encoder(dates)
        else:
            # If no dates provided, create dummy date sequence
            dummy_dates = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            calendar_emb = self.calendar_encoder(dummy_dates)
        
        # Combine positional and calendar embeddings
        combined_temporal = torch.cat([pos_emb, calendar_emb], dim=-1)
        enhanced_temporal = self.temporal_fusion(combined_temporal)
        
        return enhanced_temporal