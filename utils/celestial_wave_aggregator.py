"""
Celestial Wave Aggregator - Maps 114 waves to 13 celestial bodies

This module aggregates the 114 wave features from comprehensive_dynamic_features_dummy.csv
into 13 celestial body representations based on astrological domain knowledge.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from layers.modular.graph.celestial_body_nodes import CelestialBody

class CelestialWaveAggregator(nn.Module):
    """
    Aggregates 114 wave features into 13 celestial body representations
    
    Uses astrological domain knowledge to map waves to celestial influences:
    - Fast waves (1-7 days) → Mercury, Moon (quick changes)
    - Medium waves (1-4 weeks) → Venus, Mars (medium-term trends)  
    - Slow waves (1-12 months) → Jupiter, Saturn (long-term cycles)
    - Very slow waves (1+ years) → Uranus, Neptune, Pluto (major shifts)
    """
    
    def __init__(self, num_input_waves: int = 118, num_celestial_bodies: int = 13):
        super().__init__()
        self.num_input_waves = num_input_waves
        self.num_celestial_bodies = num_celestial_bodies
        
        # Create wave-to-celestial mapping based on astrological principles
        self.wave_mapping = self._create_astrological_mapping()
        
        # Learnable aggregation weights for each celestial body
        self.aggregation_weights = nn.ParameterDict({
            body.value: nn.Parameter(torch.ones(len(waves)) / len(waves))
            for body, waves in self.wave_mapping.items()
        })
        
        # Celestial body transformation networks
        self.celestial_transforms = nn.ModuleDict({
            body.value: nn.Sequential(
                nn.Linear(1, 32),
                nn.GELU(),
                nn.Linear(32, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Tanh()  # Bounded output
            ) for body in CelestialBody
        })
        
        print(f"🌌 Celestial Wave Aggregator initialized:")
        print(f"   - Input waves: {num_input_waves}")
        print(f"   - Celestial bodies: {num_celestial_bodies}")
        for body, waves in self.wave_mapping.items():
            print(f"   - {body.value}: {len(waves)} waves")
    
    def _create_astrological_mapping(self) -> Dict[CelestialBody, List[int]]:
        """Create mapping from waves to celestial bodies based on astrological principles"""
        
        # Divide 118 waves into groups based on time scales and astrological associations
        mapping = {}
        
        # Skip OHLC (first 4 features) and time_delta (5th feature) - start from wave 5
        wave_start = 5  # Skip log_Open, log_High, log_Low, log_Close, time_delta
        
        # Fast-moving celestial bodies (short-term influences)
        mapping[CelestialBody.MOON] = list(range(wave_start, wave_start + 9))      # Waves 5-13: Daily cycles, emotions
        mapping[CelestialBody.MERCURY] = list(range(wave_start + 9, wave_start + 18))   # Waves 14-22: Communication, quick trades
        
        # Personal planets (medium-term influences)  
        mapping[CelestialBody.VENUS] = list(range(wave_start + 18, wave_start + 27))    # Waves 23-31: Value, harmony
        mapping[CelestialBody.SUN] = list(range(wave_start + 27, wave_start + 36))      # Waves 32-40: Core trends, leadership
        mapping[CelestialBody.MARS] = list(range(wave_start + 36, wave_start + 45))     # Waves 41-49: Energy, volatility
        
        # Social planets (longer-term cycles)
        mapping[CelestialBody.JUPITER] = list(range(wave_start + 45, wave_start + 54))  # Waves 50-58: Expansion, optimism
        mapping[CelestialBody.SATURN] = list(range(wave_start + 54, wave_start + 63))   # Waves 59-67: Structure, discipline
        
        # Outer planets (generational influences)
        mapping[CelestialBody.URANUS] = list(range(wave_start + 63, wave_start + 72))   # Waves 68-76: Innovation, disruption
        mapping[CelestialBody.NEPTUNE] = list(range(wave_start + 72, wave_start + 81))  # Waves 77-85: Illusion, speculation
        mapping[CelestialBody.PLUTO] = list(range(wave_start + 81, wave_start + 90))    # Waves 86-94: Transformation, power
        
        # Lunar nodes (karmic influences)
        mapping[CelestialBody.NORTH_NODE] = list(range(wave_start + 90, wave_start + 102))  # Waves 95-106: Future trends
        mapping[CelestialBody.SOUTH_NODE] = list(range(wave_start + 102, wave_start + 110)) # Waves 107-114: Past patterns
        
        # Healing and recovery (remaining waves)
        mapping[CelestialBody.CHIRON] = list(range(wave_start + 110, 118))     # Waves 115-117: Healing, recovery
        
        return mapping
    
    def forward(self, wave_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Aggregate wave features into celestial body representations
        
        Args:
            wave_features: [batch_size, seq_len, 114] Wave features
            
        Returns:
            Tuple of:
            - celestial_features: [batch_size, seq_len, 13] Celestial body features
            - metadata: Dict with aggregation information
        """
        batch_size, seq_len, num_waves = wave_features.shape
        
        if num_waves != self.num_input_waves:
            raise ValueError(f"Expected {self.num_input_waves} waves, got {num_waves}")
        
        celestial_features = []
        aggregation_info = {}
        
        # Aggregate waves for each celestial body
        for i, body in enumerate(CelestialBody):
            wave_indices = self.wave_mapping[body]
            
            # Extract relevant waves
            body_waves = wave_features[:, :, wave_indices]  # [batch, seq_len, num_body_waves]
            
            # Weighted aggregation
            weights = torch.softmax(self.aggregation_weights[body.value], dim=0)
            aggregated = torch.sum(body_waves * weights.unsqueeze(0).unsqueeze(0), dim=-1, keepdim=True)
            # [batch, seq_len, 1]
            
            # Apply celestial transformation
            transformed = self.celestial_transforms[body.value](aggregated)
            # [batch, seq_len, 1]
            
            celestial_features.append(transformed)
            
            # Store aggregation info
            aggregation_info[body.value] = {
                'wave_indices': wave_indices,
                'weights': weights.detach(),
                'contribution': aggregated.abs().mean().item()
            }
        
        # Stack celestial features
        celestial_tensor = torch.cat(celestial_features, dim=-1)  # [batch, seq_len, 13]
        
        # Metadata
        metadata = {
            'aggregation_info': aggregation_info,
            'most_active_body': self._find_most_active_body(aggregation_info),
            'total_energy': celestial_tensor.abs().sum().item(),
            'celestial_balance': self._compute_celestial_balance(celestial_tensor)
        }
        
        return celestial_tensor, metadata
    
    def _find_most_active_body(self, aggregation_info: Dict) -> str:
        """Find the most active celestial body"""
        max_contribution = 0.0
        most_active = 'sun'  # Default
        
        for body, info in aggregation_info.items():
            if info['contribution'] > max_contribution:
                max_contribution = info['contribution']
                most_active = body
        
        return most_active
    
    def _compute_celestial_balance(self, celestial_tensor: torch.Tensor) -> Dict[str, float]:
        """Compute balance between different celestial influences"""
        
        # Average across batch and sequence dimensions
        avg_features = celestial_tensor.mean(dim=(0, 1))  # [13]
        
        # Group by astrological categories
        personal_planets = avg_features[[0, 1, 2, 3, 4]].mean().item()  # Sun, Moon, Mercury, Venus, Mars
        social_planets = avg_features[[5, 6]].mean().item()             # Jupiter, Saturn
        outer_planets = avg_features[[7, 8, 9]].mean().item()           # Uranus, Neptune, Pluto
        lunar_nodes = avg_features[[10, 11]].mean().item()              # North Node, South Node
        healing = avg_features[12].item()                               # Chiron
        
        return {
            'personal_planets': personal_planets,
            'social_planets': social_planets,
            'outer_planets': outer_planets,
            'lunar_nodes': lunar_nodes,
            'healing': healing
        }
    
    def get_wave_mapping_info(self) -> Dict[str, List[int]]:
        """Get human-readable wave mapping information"""
        return {body.value: waves for body, waves in self.wave_mapping.items()}
    
    def extract_target_waves(self, wave_features: torch.Tensor, target_indices: List[int]) -> torch.Tensor:
        """
        Extract target waves (e.g., OHLC) from the full wave features
        
        Args:
            wave_features: [batch_size, seq_len, 114] Full wave features
            target_indices: List of indices for target waves (e.g., [0, 1, 2, 3] for OHLC)
            
        Returns:
            target_waves: [batch_size, seq_len, len(target_indices)] Target wave features
        """
        return wave_features[:, :, target_indices]


class CelestialDataProcessor:
    """
    Data processor that handles the wave-to-celestial aggregation for training
    """
    
    def __init__(self, aggregator: CelestialWaveAggregator, target_indices: List[int] = [0, 1, 2, 3]):
        self.aggregator = aggregator
        self.target_indices = target_indices
    
    def process_batch(self, batch_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Process a batch of wave data into celestial nodes and targets
        
        Args:
            batch_data: [batch_size, seq_len, 114] Wave features
            
        Returns:
            Tuple of:
            - celestial_nodes: [batch_size, seq_len, 13] Celestial body features for graph nodes
            - target_waves: [batch_size, seq_len, 4] Target waves (OHLC) for prediction
            - metadata: Processing metadata
        """
        # Aggregate waves to celestial bodies (for graph nodes)
        celestial_nodes, celestial_metadata = self.aggregator(batch_data)
        
        # Extract target waves (for prediction targets)
        target_waves = self.aggregator.extract_target_waves(batch_data, self.target_indices)
        
        metadata = {
            'celestial_metadata': celestial_metadata,
            'target_indices': self.target_indices,
            'input_shape': batch_data.shape,
            'celestial_shape': celestial_nodes.shape,
            'target_shape': target_waves.shape
        }
        
        return celestial_nodes, target_waves, metadata