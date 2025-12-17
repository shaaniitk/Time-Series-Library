"""
Celestial Body Graph Nodes - Astronomical Domain Knowledge for Financial Markets

This module implements a revolutionary approach to financial time series modeling by representing
market influences as celestial bodies with learned relationships based on astrological aspects.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math

class CelestialBody(Enum):
    """Celestial bodies with their traditional astrological associations"""
    SUN = "sun"           # Core vitality, leadership, gold markets
    MOON = "moon"         # Emotions, cycles, silver markets  
    MERCURY = "mercury"   # Communication, technology, quick trades
    VENUS = "venus"       # Values, luxury goods, currencies
    MARS = "mars"         # Energy, conflict, volatility
    JUPITER = "jupiter"   # Expansion, optimism, bull markets
    SATURN = "saturn"     # Structure, discipline, bear markets
    URANUS = "uranus"     # Innovation, disruption, crypto
    NEPTUNE = "neptune"   # Illusion, speculation, bubbles
    PLUTO = "pluto"       # Transformation, power, major shifts
    NORTH_NODE = "north_node"  # Future trends, growth direction
    SOUTH_NODE = "south_node"  # Past patterns, corrections
    CHIRON = "chiron"     # Healing, recovery, market wounds

class AspectType(Enum):
    """Astrological aspects representing relationship types"""
    CONJUNCTION = 0      # 0° - Unity, amplification
    SEXTILE = 60        # 60° - Opportunity, harmony
    SQUARE = 90         # 90° - Tension, volatility
    TRINE = 120         # 120° - Flow, ease
    OPPOSITION = 180    # 180° - Polarity, extremes

class CelestialBodyNodes(nn.Module):
    """
    Celestial Body Graph Nodes with Astronomical Domain Knowledge
    
    Creates semantic nodes representing celestial bodies instead of raw parameters,
    enabling interpretable astrological AI for financial markets.
    """
    
    def __init__(self, d_model: int = 512, num_aspects: int = 5):
        super().__init__()
        self.d_model = d_model
        self.num_bodies = len(CelestialBody)
        self.num_aspects = num_aspects
        
        # Celestial body embeddings - learnable representations
        self.body_embeddings = nn.Parameter(
            torch.randn(self.num_bodies, d_model) * 0.02
        )
        
        # Aspect embeddings - relationship types
        self.aspect_embeddings = nn.Parameter(
            torch.randn(self.num_aspects, d_model) * 0.02
        )
        
        # Body-specific transformation networks
        self.body_transforms = nn.ModuleDict({
            body.value: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model)
            ) for body in CelestialBody
        })
        
        # Aspect strength calculator
        self.aspect_strength_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Market condition encoder
        self.market_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Initialize with astronomical knowledge
        self._initialize_astronomical_knowledge()
    
    def _initialize_astronomical_knowledge(self):
        """Initialize embeddings with astronomical domain knowledge"""
        
        # Traditional astrological associations (normalized)
        body_init_values = {
            CelestialBody.SUN: [1.0, 0.8, 0.9, 0.7],      # Leadership, vitality
            CelestialBody.MOON: [0.3, 0.9, 0.4, 0.8],     # Emotion, cycles
            CelestialBody.MERCURY: [0.9, 0.3, 0.8, 0.4],  # Speed, communication
            CelestialBody.VENUS: [0.6, 0.7, 0.8, 0.9],    # Value, harmony
            CelestialBody.MARS: [0.9, 0.2, 0.3, 0.8],     # Energy, conflict
            CelestialBody.JUPITER: [0.8, 0.9, 0.7, 0.6],  # Expansion, optimism
            CelestialBody.SATURN: [0.2, 0.8, 0.9, 0.3],   # Structure, discipline
            CelestialBody.URANUS: [0.9, 0.1, 0.2, 0.9],   # Innovation, disruption
            CelestialBody.NEPTUNE: [0.1, 0.9, 0.8, 0.2],  # Illusion, speculation
            CelestialBody.PLUTO: [0.8, 0.1, 0.9, 0.9],    # Transformation, power
            CelestialBody.NORTH_NODE: [0.7, 0.6, 0.8, 0.5], # Future trends
            CelestialBody.SOUTH_NODE: [0.3, 0.4, 0.2, 0.5], # Past patterns
            CelestialBody.CHIRON: [0.4, 0.6, 0.5, 0.7]    # Healing, recovery
        }
        
        with torch.no_grad():
            for i, body in enumerate(CelestialBody):
                if body in body_init_values:
                    # Expand to full d_model dimensions
                    init_pattern = body_init_values[body]
                    full_init = []
                    for j in range(self.d_model):
                        full_init.append(init_pattern[j % len(init_pattern)])
                    
                    self.body_embeddings[i] = torch.tensor(full_init, dtype=torch.float32) * 0.1
    
    def get_astronomical_adjacency(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate astronomical adjacency matrix and broadcast it to be dynamic.
        
        Returns:
            torch.Tensor: [batch_size, seq_len, num_bodies, num_bodies] adjacency matrix
        """
        
        # Fixed astronomical relationships (can be made learnable later)
        astronomical_connections = {
            # Major harmonious aspects (Trines - 120°)
            (CelestialBody.SUN, CelestialBody.JUPITER): 0.9,    # Success, leadership
            (CelestialBody.MOON, CelestialBody.VENUS): 0.8,     # Emotional value
            (CelestialBody.MERCURY, CelestialBody.URANUS): 0.7, # Innovation communication
            
            # Challenging aspects (Squares - 90°)  
            (CelestialBody.MARS, CelestialBody.SATURN): -0.6,   # Energy vs structure
            (CelestialBody.SUN, CelestialBody.PLUTO): -0.5,     # Power struggles
            (CelestialBody.JUPITER, CelestialBody.NEPTUNE): -0.4, # Overoptimism
            
            # Oppositions (180°) - Polarity
            (CelestialBody.NORTH_NODE, CelestialBody.SOUTH_NODE): -0.8, # Past vs future
            (CelestialBody.SUN, CelestialBody.SATURN): -0.7,    # Growth vs restriction
            
            # Conjunctions (0°) - Unity
            (CelestialBody.VENUS, CelestialBody.JUPITER): 0.9,  # Value expansion
            (CelestialBody.MERCURY, CelestialBody.MARS): 0.6,   # Quick action
            
            # Sextiles (60°) - Opportunity
            (CelestialBody.MOON, CelestialBody.MERCURY): 0.5,   # Intuitive communication
            (CelestialBody.VENUS, CelestialBody.MARS): 0.4,     # Balanced energy
        }
        
        # Create adjacency matrix
        adj_matrix = torch.zeros(self.num_bodies, self.num_bodies, device=device)
        
        for (body1, body2), strength in astronomical_connections.items():
            i = list(CelestialBody).index(body1)
            j = list(CelestialBody).index(body2)
            adj_matrix[i, j] = strength
            adj_matrix[j, i] = strength  # Symmetric
        
        # Add self-connections (each body influences itself)
        adj_matrix.fill_diagonal_(1.0)
        
        # Expand to batch and sequence dimensions
        return adj_matrix.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
    
    def compute_dynamic_aspects(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic aspect strengths for each time step based on encoder output.
        Optimized with vectorization to avoid O(N^2) Python loops.
        """
        batch_size, seq_len, _ = enc_out.shape
        device = enc_out.device
        
        # Encode market conditions: [batch*seq_len, d_model]
        enc_out_flat = enc_out.reshape(batch_size * seq_len, self.d_model)
        market_encoded = self.market_encoder(enc_out_flat)
        
        # Prepare context-aware body embeddings using broadcasting
        # body_embeddings: [num_bodies, d_model] -> [1, num_bodies, d_model]
        # market_encoded: [B*S, d_model] -> [B*S, 1, d_model]
        # Result: [B*S, num_bodies, d_model]
        contextualized_bodies = self.body_embeddings.unsqueeze(0) + market_encoded.unsqueeze(1)
        
        # Generate indices for upper triangle pairs (i < j)
        rows, cols = torch.triu_indices(self.num_bodies, self.num_bodies, offset=1, device=device)
        
        # Gather embeddings for all pairs in parallel
        # body_i_all: [B*S, num_pairs, d_model]
        body_i_all = contextualized_bodies[:, rows, :]
        body_j_all = contextualized_bodies[:, cols, :]
        
        # Create pairwise input for the network
        # [B*S, num_pairs, 2*d_model]
        aspect_input = torch.cat([body_i_all, body_j_all], dim=-1)
        
        # Compute strengths for all pairs in one heavy batch operation
        # [B*S, num_pairs, 1] -> squeeze -> [B*S, num_pairs]
        strengths = self.aspect_strength_net(aspect_input).squeeze(-1)
        
        # Map back to adjacency matrix structure
        dynamic_adj_flat = torch.zeros(batch_size * seq_len, self.num_bodies, self.num_bodies, device=device)
        
        # Fill symmetric pairs
        dynamic_adj_flat[:, rows, cols] = strengths
        dynamic_adj_flat[:, cols, rows] = strengths
        
        # Fill diagonal with 1.0 (self-influence)
        dynamic_adj_flat.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        
        # Reshape to 4D tensor: [batch, seq_len, num_bodies, num_bodies]
        return dynamic_adj_flat.view(batch_size, seq_len, self.num_bodies, self.num_bodies)
    
    def get_celestial_features(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        Get transformed celestial body features for each time step.
        
        Args:
            enc_out: [batch_size, seq_len, d_model] full encoder output
            
        Returns:
            torch.Tensor: [batch_size, seq_len, num_bodies, d_model] celestial features
        """
        batch_size, seq_len, _ = enc_out.shape
        enc_out_flat = enc_out.reshape(batch_size * seq_len, self.d_model)
        
        # Transform each celestial body based on market context at each time step
        celestial_features_flat = []
        
        for i, body in enumerate(CelestialBody):
            # Get base embedding
            base_embedding = self.body_embeddings[i]  # [d_model]
            
            # Add market context from each time step
            contextualized = base_embedding + enc_out_flat  # [batch*seq_len, d_model]
            
            # Apply body-specific transformation
            transformed = self.body_transforms[body.value](contextualized)  # [batch*seq_len, d_model]
            
            celestial_features_flat.append(transformed)
        
        # Stack and reshape
        stacked_features = torch.stack(celestial_features_flat, dim=1)  # [batch*seq_len, num_bodies, d_model]
        return stacked_features.view(batch_size, seq_len, self.num_bodies, self.d_model)
    
    def forward(self, enc_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass to generate dynamic celestial body graphs and features.
        
        Args:
            enc_out: [batch_size, seq_len, d_model] full encoder output
            
        Returns:
            Tuple containing:
            - astronomical_adj: [batch_size, seq_len, num_bodies, num_bodies] broadcasted astronomical adjacency
            - dynamic_adj: [batch_size, seq_len, num_bodies, num_bodies] learned dynamic adjacency  
            - celestial_features: [batch_size, seq_len, num_bodies, d_model] dynamic body features
            - metadata: Dict with interpretability information
        """
        batch_size, seq_len, _ = enc_out.shape
        device = enc_out.device
        
        # Get fixed astronomical adjacency, broadcasted to be dynamic
        astronomical_adj = self.get_astronomical_adjacency(batch_size, seq_len, device)
        
        # Compute dynamic aspects based on the full sequence
        dynamic_adj = self.compute_dynamic_aspects(enc_out)
        
        # Get dynamic celestial body features
        celestial_features = self.get_celestial_features(enc_out)
        
        # Metadata for interpretability (averaged over sequence for stability)
        metadata = {
            'body_names': [body.value for body in CelestialBody],
            'astronomical_strength': astronomical_adj.abs().mean().item(),
            'dynamic_strength': dynamic_adj.abs().mean().item(),
            'most_active_body': torch.argmax(celestial_features.norm(dim=-1).mean(dim=[0, 1])).item()
        }
        
        return astronomical_adj, dynamic_adj, celestial_features, metadata

    def get_body_interpretation(self, body_idx: int) -> Dict[str, str]:
        """Get human-readable interpretation of a celestial body"""
        
        body = list(CelestialBody)[body_idx]
        
        interpretations = {
            CelestialBody.SUN: {
                'name': 'Sun',
                'domain': 'Leadership & Core Trends',
                'market_influence': 'Major market direction, gold prices, leadership stocks'
            },
            CelestialBody.MOON: {
                'name': 'Moon', 
                'domain': 'Emotions & Cycles',
                'market_influence': 'Market sentiment, cyclical patterns, silver prices'
            },
            CelestialBody.MERCURY: {
                'name': 'Mercury',
                'domain': 'Communication & Speed', 
                'market_influence': 'Tech stocks, news-driven moves, high-frequency trading'
            },
            CelestialBody.VENUS: {
                'name': 'Venus',
                'domain': 'Value & Harmony',
                'market_influence': 'Currency markets, luxury goods, stable value plays'
            },
            CelestialBody.MARS: {
                'name': 'Mars',
                'domain': 'Energy & Conflict',
                'market_influence': 'Volatility spikes, energy sector, aggressive moves'
            },
            CelestialBody.JUPITER: {
                'name': 'Jupiter', 
                'domain': 'Expansion & Optimism',
                'market_influence': 'Bull markets, growth stocks, optimistic sentiment'
            },
            CelestialBody.SATURN: {
                'name': 'Saturn',
                'domain': 'Structure & Discipline', 
                'market_influence': 'Bear markets, regulations, conservative plays'
            },
            CelestialBody.URANUS: {
                'name': 'Uranus',
                'domain': 'Innovation & Disruption',
                'market_influence': 'Crypto markets, disruptive tech, sudden changes'
            },
            CelestialBody.NEPTUNE: {
                'name': 'Neptune',
                'domain': 'Illusion & Speculation',
                'market_influence': 'Bubbles, speculation, unclear market conditions'
            },
            CelestialBody.PLUTO: {
                'name': 'Pluto',
                'domain': 'Transformation & Power',
                'market_influence': 'Major market shifts, power consolidation, deep changes'
            },
            CelestialBody.NORTH_NODE: {
                'name': 'North Node',
                'domain': 'Future Trends',
                'market_influence': 'Emerging trends, growth direction, future opportunities'
            },
            CelestialBody.SOUTH_NODE: {
                'name': 'South Node', 
                'domain': 'Past Patterns',
                'market_influence': 'Historical patterns, corrections, past influences'
            },
            CelestialBody.CHIRON: {
                'name': 'Chiron',
                'domain': 'Healing & Recovery',
                'market_influence': 'Market recovery, healing from crashes, wounded sectors'
            }
        }
        
        return interpretations.get(body, {'name': 'Unknown', 'domain': 'Unknown', 'market_influence': 'Unknown'})