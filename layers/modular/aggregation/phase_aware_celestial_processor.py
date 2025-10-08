"""
Phase-Aware Celestial Processor

This processor is optimized for inputs that already contain phase information
(sin/cos encoded), velocity, radius, and longitude for each celestial body.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from layers.modular.graph.celestial_body_nodes import CelestialBody


class PhaseAwareCelestialProcessor(nn.Module):
    """
    Celestial processor optimized for phase-rich inputs.
    
    Assumes input structure per celestial body:
    - sin(θ), cos(θ) for phase θ
    - sin(φ), cos(φ) for phase φ  
    - velocity
    - r (radius)
    - longitude
    - additional features...
    
    Key features:
    1. Preserves phase information (no re-computation needed)
    2. Explicitly computes phase differences for edges
    3. Models phase shift relationships between celestial bodies
    4. Rich multi-dimensional celestial representations
    """
    
    def __init__(self, num_input_waves: int = 118, celestial_dim: int = 32, 
                 waves_per_body: int = 9, num_heads: int = 8):
        super().__init__()
        self.num_input_waves = num_input_waves
        self.celestial_dim = celestial_dim
        self.waves_per_body = waves_per_body
        self.num_heads = num_heads
        
        # Wave-to-celestial mapping
        self.wave_mapping = self._create_astrological_mapping()
        
        # For each celestial body, create a processor that understands phase structure
        self.celestial_processors = nn.ModuleDict()
        for body in CelestialBody:
            num_body_waves = len(self.wave_mapping[body])
            self.celestial_processors[body.value] = PhaseAwareCelestialBodyProcessor(
                num_input_features=num_body_waves,
                output_dim=celestial_dim,
                num_heads=min(num_heads, 4),
                celestial_body_name=body.value
            )
        
        # Phase difference computer for edges
        self.phase_difference_computer = PhaseDifferenceEdgeComputer(
            celestial_dim=celestial_dim,
            num_celestial_bodies=len(CelestialBody)
        )
        
        # Inter-celestial attention with phase awareness
        self.inter_celestial_attention = PhaseAwareAttention(
            embed_dim=celestial_dim,
            num_heads=num_heads
        )
        
        print(f"🌌 Phase-Aware Celestial Processor initialized:")
        print(f"   - Input waves: {num_input_waves}")
        print(f"   - Celestial bodies: {len(CelestialBody)}")
        print(f"   - Celestial dimension: {celestial_dim}")
        print(f"   - Assumes phase info already in inputs (sin/cos encoded)")
        print(f"   - Will compute explicit phase differences for edges")
    
    def _create_astrological_mapping(self) -> Dict[CelestialBody, List[int]]:
        """
        Create mapping from actual input features to celestial bodies.
        
        Based on analysis of prepared_financial_data.csv:
        - Features 0-4: OHLC + time_delta (excluded)
        - Features 5-91: Dynamic celestial features (7 per body for most)
        - Features 92-98: Shadbala strength features  
        - Features 99-120: Static celestial features (2 per body)
        """
        mapping = {}
        
        # Dynamic features start at index 5 (after OHLC + time_delta)
        # Each celestial body has: sin, cos, speed, sign_sin, sign_cos, distance, lat
        
        # Create safe mapping that stays within bounds
        features_per_body = 9
        start_idx = 5  # Skip OHLC + time_delta
        
        bodies = [CelestialBody.SUN, CelestialBody.MOON, CelestialBody.MARS, 
                 CelestialBody.MERCURY, CelestialBody.JUPITER, CelestialBody.VENUS,
                 CelestialBody.SATURN, CelestialBody.URANUS, CelestialBody.NEPTUNE,
                 CelestialBody.PLUTO, CelestialBody.NORTH_NODE, CelestialBody.SOUTH_NODE]
        
        for i, body in enumerate(bodies):
            start = start_idx + i * features_per_body
            end = min(start + features_per_body, self.num_input_waves)
            if start < self.num_input_waves:
                mapping[body] = list(range(start, end))
            else:
                # If we run out of features, use the last available ones
                mapping[body] = list(range(max(5, self.num_input_waves - features_per_body), self.num_input_waves))
        
        # CHIRON - use remaining features or duplicate some
        chiron_start = start_idx + len(bodies) * features_per_body
        if chiron_start < self.num_input_waves:
            mapping[CelestialBody.CHIRON] = list(range(chiron_start, min(chiron_start + 3, self.num_input_waves)))
        else:
            # Use some features from the end
            mapping[CelestialBody.CHIRON] = list(range(max(5, self.num_input_waves - 3), self.num_input_waves))
        
        return mapping
    
    def forward(self, wave_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Process phase-rich wave features.
        
        Args:
            wave_features: [batch, seq_len, 118] Wave features with embedded phase info
            
        Returns:
            Tuple of:
            - celestial_features: [batch, seq_len, 13 * celestial_dim] Rich celestial representations
            - adjacency_matrix: [batch, 13, 13] Phase-difference based edges
            - metadata: Rich metadata including phase analysis
        """
        batch_size, seq_len, num_waves = wave_features.shape
        
        # Process each celestial body
        celestial_representations = []
        celestial_metadata = {}
        extracted_phases = {}  # Store extracted phase info for edge computation
        
        for body in CelestialBody:
            wave_indices = self.wave_mapping[body]
            body_waves = wave_features[:, :, wave_indices]  # [batch, seq_len, num_body_waves]
            
            # Process with phase awareness
            celestial_repr, phase_info, body_meta = self.celestial_processors[body.value](body_waves)
            # celestial_repr: [batch, seq_len, celestial_dim]
            
            celestial_representations.append(celestial_repr)
            celestial_metadata[body.value] = body_meta
            extracted_phases[body.value] = phase_info
        
        # Stack celestial representations
        celestial_tensor = torch.stack(celestial_representations, dim=2)  # [batch, seq_len, 13, celestial_dim]
        celestial_flat = celestial_tensor.view(batch_size, seq_len, -1)  # [batch, seq_len, 13 * celestial_dim]
        
        # Compute phase-difference based edges
        adjacency_matrix, edge_metadata = self.phase_difference_computer(
            celestial_tensor, extracted_phases
        )
        
        # Apply inter-celestial attention with phase awareness
        enhanced_celestial = self.inter_celestial_attention(
            celestial_tensor, adjacency_matrix
        )
        enhanced_flat = enhanced_celestial.reshape(batch_size, seq_len, -1)
        
        # Comprehensive metadata
        metadata = {
            'celestial_metadata': celestial_metadata,
            'phase_info': extracted_phases,
            'edge_metadata': edge_metadata,
            'phase_coherence': self._compute_global_phase_coherence(extracted_phases),
            'celestial_energy': enhanced_flat.abs().mean().item()
        }
        
        return enhanced_flat, adjacency_matrix, metadata
    
    def _compute_global_phase_coherence(self, phase_info: Dict) -> Dict:
        """Compute global phase coherence across all celestial bodies."""
        all_theta_phases = []
        all_phi_phases = []
        
        for body_phases in phase_info.values():
            if 'theta_phase' in body_phases:
                all_theta_phases.append(body_phases['theta_phase'])
            if 'phi_phase' in body_phases:
                all_phi_phases.append(body_phases['phi_phase'])
        
        coherence_info = {}
        
        if all_theta_phases:
            theta_stack = torch.stack(all_theta_phases, dim=-1)  # [batch, seq_len, num_bodies]
            theta_coherence = self._phase_coherence(theta_stack)
            coherence_info['theta_coherence'] = theta_coherence.mean().item()
        
        if all_phi_phases:
            phi_stack = torch.stack(all_phi_phases, dim=-1)
            phi_coherence = self._phase_coherence(phi_stack)
            coherence_info['phi_coherence'] = phi_coherence.mean().item()
        
        return coherence_info
    
    def _phase_coherence(self, phases: torch.Tensor) -> torch.Tensor:
        """Compute phase coherence using circular statistics."""
        # Convert to complex exponentials
        complex_phases = torch.exp(1j * phases)
        
        # Mean resultant vector
        mean_vector = complex_phases.mean(dim=-1)
        
        # Coherence is magnitude of mean vector
        coherence = torch.abs(mean_vector)
        
        return coherence.real


class PhaseAwareCelestialBodyProcessor(nn.Module):
    """
    Process a single celestial body's features with phase awareness.
    
    Assumes input features include sin/cos encoded phases, velocity, radius, etc.
    """
    
    def __init__(self, num_input_features: int, output_dim: int, num_heads: int = 4, celestial_body_name: str = 'Sun'):
        super().__init__()
        self.num_input_features = num_input_features
        self.output_dim = output_dim
        self.celestial_body_name = celestial_body_name
        
        # Phase extraction based on celestial body type
        self.phase_extractor = PhaseExtractor(celestial_body_name)
        
        # Feature processing with attention to different types
        # Ensure embed_dim is divisible by num_heads
        effective_heads = min(num_heads, num_input_features)
        while num_input_features % effective_heads != 0 and effective_heads > 1:
            effective_heads -= 1
        
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=num_input_features,
            num_heads=effective_heads,
            batch_first=True
        )
        
        # Rich feature transformation
        self.feature_transformer = nn.Sequential(
            nn.Linear(num_input_features, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Temporal dynamics modeling
        self.temporal_processor = nn.GRU(
            input_size=output_dim,
            hidden_size=output_dim,
            batch_first=True
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Args:
            features: [batch, seq_len, num_input_features]
        Returns:
            Tuple of (celestial_representation, phase_info, metadata)
        """
        batch_size, seq_len, num_features = features.shape
        
        # Extract phase information
        phase_info = self.phase_extractor(features)
        
        # Apply feature attention
        features_flat = features.view(batch_size * seq_len, 1, num_features)
        attended_features, _ = self.feature_attention(features_flat, features_flat, features_flat)
        attended_features = attended_features.view(batch_size, seq_len, num_features)
        
        # Transform to celestial representation
        celestial_repr = self.feature_transformer(attended_features)
        
        # Add temporal dynamics
        temporal_output, _ = self.temporal_processor(celestial_repr)
        
        # Combine with residual connection
        final_repr = celestial_repr + temporal_output
        
        metadata = {
            'feature_energy': attended_features.abs().mean().item(),
            'celestial_energy': final_repr.abs().mean().item(),
            'temporal_contribution': temporal_output.abs().mean().item()
        }
        
        return final_repr, phase_info, metadata


class PhaseExtractor(nn.Module):
    """Extract phase information from actual celestial features."""
    
    def __init__(self, celestial_body: str):
        super().__init__()
        self.celestial_body = celestial_body
        
        # Define feature structure based on celestial body type
        self.has_shadbala = celestial_body in ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn']
        self.has_distance_lat = celestial_body not in ['North_Node', 'South_Node']
        
    def forward(self, features: torch.Tensor) -> Dict:
        """
        Extract phase information from celestial body features.
        
        Feature structure for most bodies:
        [sin_longitude, cos_longitude, speed, sign_sin, sign_cos, distance, lat, shadbala, static_sin, static_cos]
        
        For outer planets (no shadbala):
        [sin_longitude, cos_longitude, speed, sign_sin, sign_cos, distance, lat, static_sin, static_cos]
        
        For nodes (no distance/lat/shadbala):
        [sin_longitude, cos_longitude, speed, sign_sin, sign_cos, static_sin, static_cos]
        
        Args:
            features: [batch, seq_len, num_features] Features for this celestial body
        Returns:
            Dictionary with extracted phase information
        """
        batch_size, seq_len, num_features = features.shape
        
        phase_info = {}
        
        # Extract longitude phase (θ) - always first two features
        sin_longitude = features[:, :, 0]
        cos_longitude = features[:, :, 1]
        longitude_phase = torch.atan2(sin_longitude, cos_longitude)
        
        phase_info['longitude_phase'] = longitude_phase  # θ phase
        phase_info['longitude_sin'] = sin_longitude
        phase_info['longitude_cos'] = cos_longitude
        
        # Extract speed/velocity - always 3rd feature
        phase_info['speed'] = features[:, :, 2]
        
        # Extract zodiac sign phase (φ) - always 4th and 5th features
        if num_features >= 5:
            sin_sign = features[:, :, 3]
            cos_sign = features[:, :, 4]
            sign_phase = torch.atan2(sin_sign, cos_sign)
            
            phase_info['sign_phase'] = sign_phase  # φ phase
            phase_info['sign_sin'] = sin_sign
            phase_info['sign_cos'] = cos_sign
        
        # Extract distance and latitude (if available)
        if self.has_distance_lat and num_features >= 7:
            phase_info['distance'] = features[:, :, 5]  # r
            phase_info['latitude'] = features[:, :, 6]
            
            # Shadbala strength (if available)
            if self.has_shadbala and num_features >= 8:
                phase_info['shadbala'] = features[:, :, 7]
                static_start_idx = 8
            else:
                static_start_idx = 7
        else:
            # For nodes - no distance/lat
            static_start_idx = 5
            if self.has_shadbala and num_features > 5:
                phase_info['shadbala'] = features[:, :, 5]
                static_start_idx = 6
        
        # Extract ecliptic longitude (fundamental astrological coordinate - 2D projection)
        if num_features >= static_start_idx + 2:
            sin_ecliptic = features[:, :, static_start_idx]
            cos_ecliptic = features[:, :, static_start_idx + 1]
            ecliptic_longitude = torch.atan2(sin_ecliptic, cos_ecliptic)
            
            phase_info['ecliptic_longitude'] = ecliptic_longitude  # Fundamental astrological angle
            phase_info['ecliptic_sin'] = sin_ecliptic
            phase_info['ecliptic_cos'] = cos_ecliptic
        
        # Compute phase velocity (temporal derivative of longitude phase)
        if seq_len > 1:
            phase_velocity = torch.zeros_like(longitude_phase)
            phase_velocity[:, 1:] = self._circular_difference(
                longitude_phase[:, 1:], longitude_phase[:, :-1]
            )
            phase_info['phase_velocity'] = phase_velocity
        else:
            phase_info['phase_velocity'] = torch.zeros_like(longitude_phase)
        
        return phase_info
    
    def _circular_difference(self, angle1: torch.Tensor, angle2: torch.Tensor) -> torch.Tensor:
        """Compute circular difference between consecutive angles."""
        diff = angle1 - angle2
        # Wrap to [-π, π]
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        return diff


class PhaseDifferenceEdgeComputer(nn.Module):
    """
    Compute edges based on explicit phase differences between celestial bodies.
    
    This answers your question: We EXPLICITLY compute phase differences for edges
    rather than relying on the network to learn them automatically.
    """
    
    def __init__(self, celestial_dim: int, num_celestial_bodies: int):
        super().__init__()
        self.celestial_dim = celestial_dim
        self.num_celestial_bodies = num_celestial_bodies
        
        # Edge strength predictor based on explicit phase differences
        self.edge_predictor = nn.Sequential(
            nn.Linear(celestial_dim * 2 + 6, celestial_dim),  # features + 6 phase difference measures
            nn.GELU(),
            nn.Linear(celestial_dim, celestial_dim // 2),
            nn.GELU(),
            nn.Linear(celestial_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Learnable phase difference weights (importance of different phase relationships)
        self.phase_weights = nn.Parameter(torch.ones(6))  # theta_diff, phi_diff, velocity_diff, etc.
    
    def forward(self, celestial_tensor: torch.Tensor, 
                phase_info: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute edges based on EXPLICIT phase differences.
        
        Args:
            celestial_tensor: [batch, seq_len, num_bodies, celestial_dim]
            phase_info: Dictionary with phase information for each celestial body
        Returns:
            Tuple of (adjacency_matrix, metadata)
        """
        batch_size, seq_len, num_bodies, celestial_dim = celestial_tensor.shape
        
        # Use last timestep for edge computation
        celestial_features = celestial_tensor[:, -1, :, :]  # [batch, num_bodies, celestial_dim]
        
        # Extract current phase information
        current_phases = {}
        body_names = list(CelestialBody)
        
        for i, body in enumerate(body_names):
            body_info = phase_info[body.value]
            
            # Helper function to safely extract last timestep
            def safe_extract_last(tensor, default_shape):
                if tensor is None:
                    return torch.zeros(default_shape, device=celestial_tensor.device)
                if tensor.dim() == 1:
                    return tensor  # Already 1D, assume it's per-batch
                elif tensor.dim() == 2:
                    return tensor[:, -1]  # Extract last timestep
                else:
                    return tensor.view(batch_size, -1)[:, -1]  # Flatten and extract last
            
            current_phases[i] = {
                'theta': safe_extract_last(body_info.get('theta_phase'), (batch_size,)),
                'phi': safe_extract_last(body_info.get('phi_phase'), (batch_size,)),
                'velocity': safe_extract_last(body_info.get('velocity'), (batch_size,)),
                'radius': safe_extract_last(body_info.get('radius'), (batch_size,)),
                'longitude': safe_extract_last(body_info.get('longitude'), (batch_size,))
            }
        
        # Create adjacency matrix with explicit phase difference computation
        adjacency = torch.zeros(batch_size, num_bodies, num_bodies, device=celestial_tensor.device)
        
        phase_diff_stats = {
            'theta_diffs': [],
            'phi_diffs': [],
            'velocity_diffs': [],
            'radius_ratios': [],
            'longitude_diffs': [],
            'edge_strengths': []
        }
        
        for i in range(num_bodies):
            for j in range(i + 1, num_bodies):
                # Celestial body features
                feat_i = celestial_features[:, i, :]
                feat_j = celestial_features[:, j, :]
                
                # EXPLICIT phase differences
                theta_diff = self._circular_difference(current_phases[i]['theta'], current_phases[j]['theta'])
                phi_diff = self._circular_difference(current_phases[i]['phi'], current_phases[j]['phi'])
                velocity_diff = current_phases[i]['velocity'] - current_phases[j]['velocity']
                radius_ratio = current_phases[i]['radius'] / (current_phases[j]['radius'] + 1e-8)
                longitude_diff = self._circular_difference(current_phases[i]['longitude'], current_phases[j]['longitude'])
                
                # Composite phase relationship measure
                phase_relationship = torch.cos(theta_diff) * torch.cos(phi_diff)  # Phase alignment measure
                
                # Combine all information for edge prediction
                edge_input = torch.cat([
                    feat_i, feat_j,  # Celestial body features
                    theta_diff.unsqueeze(-1),
                    phi_diff.unsqueeze(-1),
                    velocity_diff.unsqueeze(-1),
                    radius_ratio.unsqueeze(-1),
                    longitude_diff.unsqueeze(-1),
                    phase_relationship.unsqueeze(-1)
                ], dim=-1)
                
                # Predict edge strength with explicit phase differences
                edge_strength = self.edge_predictor(edge_input).squeeze(-1)
                
                # Apply learnable phase difference weighting
                phase_features = torch.stack([
                    theta_diff.abs(), phi_diff.abs(), velocity_diff.abs(),
                    (radius_ratio - 1).abs(), longitude_diff.abs(), phase_relationship.abs()
                ], dim=-1)
                
                weighted_phase_importance = torch.sum(phase_features * self.phase_weights.unsqueeze(0), dim=-1)
                final_edge_strength = edge_strength * torch.sigmoid(weighted_phase_importance)
                
                # Store in adjacency matrix (symmetric)
                adjacency[:, i, j] = final_edge_strength
                adjacency[:, j, i] = final_edge_strength
                
                # Collect statistics
                phase_diff_stats['theta_diffs'].append(theta_diff.abs().mean().item())
                phase_diff_stats['phi_diffs'].append(phi_diff.abs().mean().item())
                phase_diff_stats['velocity_diffs'].append(velocity_diff.abs().mean().item())
                phase_diff_stats['radius_ratios'].append(radius_ratio.mean().item())
                phase_diff_stats['longitude_diffs'].append(longitude_diff.abs().mean().item())
                phase_diff_stats['edge_strengths'].append(final_edge_strength.mean().item())
        
        # Compute metadata
        metadata = {
            'avg_theta_diff': torch.mean(torch.tensor(phase_diff_stats['theta_diffs'])).item(),
            'avg_phi_diff': torch.mean(torch.tensor(phase_diff_stats['phi_diffs'])).item(),
            'avg_velocity_diff': torch.mean(torch.tensor(phase_diff_stats['velocity_diffs'])).item(),
            'avg_radius_ratio': torch.mean(torch.tensor(phase_diff_stats['radius_ratios'])).item(),
            'avg_longitude_diff': torch.mean(torch.tensor(phase_diff_stats['longitude_diffs'])).item(),
            'avg_edge_strength': torch.mean(torch.tensor(phase_diff_stats['edge_strengths'])).item(),
            'edge_density': (adjacency > 0.5).float().mean().item(),
            'phase_weights': self.phase_weights.detach().cpu().numpy().tolist(),
            'strong_phase_relationships': (adjacency > 0.8).sum().item(),
            'weak_phase_relationships': (adjacency < 0.2).sum().item()
        }
        
        return adjacency, metadata
    
    def _circular_difference(self, angle1: torch.Tensor, angle2: torch.Tensor) -> torch.Tensor:
        """Compute circular difference between two angles."""
        diff = angle1 - angle2
        # Wrap to [-π, π]
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        return diff


class PhaseAwareAttention(nn.Module):
    """Attention mechanism that considers phase relationships."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Phase-aware attention scaling
        self.phase_scaler = nn.Parameter(torch.ones(1))
    
    def forward(self, celestial_tensor: torch.Tensor, 
                adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply phase-aware attention.
        
        Args:
            celestial_tensor: [batch, seq_len, num_bodies, embed_dim]
            adjacency_matrix: [batch, num_bodies, num_bodies]
        Returns:
            Enhanced celestial tensor
        """
        batch_size, seq_len, num_bodies, embed_dim = celestial_tensor.shape
        
        # Reshape for attention
        celestial_flat = celestial_tensor.view(batch_size * seq_len, num_bodies, embed_dim)
        
        # Create attention mask from adjacency matrix
        # Use adjacency as attention bias (higher adjacency = more attention)
        adjacency_expanded = adjacency_matrix.unsqueeze(1).expand(-1, seq_len, -1, -1)
        adjacency_flat = adjacency_expanded.reshape(batch_size * seq_len, num_bodies, num_bodies)
        
        # Convert adjacency to attention bias (log space for numerical stability)
        attention_bias = torch.log(adjacency_flat + 1e-8) * self.phase_scaler
        
        # Apply attention with phase-aware bias
        attended_output, _ = self.attention(
            celestial_flat, celestial_flat, celestial_flat,
            attn_mask=None  # We use bias instead of mask
        )
        
        # Apply phase-aware scaling
        phase_scaled_output = attended_output * (1 + attention_bias.mean(dim=-1, keepdim=True))
        
        # Reshape back
        output = phase_scaled_output.view(batch_size, seq_len, num_bodies, embed_dim)
        
        return output