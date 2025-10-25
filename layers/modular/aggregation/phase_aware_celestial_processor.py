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
        
        # Auto-detect correct number of input waves from CSV
        self.num_input_waves = self._auto_detect_input_waves(num_input_waves)
        self.celestial_dim = celestial_dim
        self.waves_per_body = waves_per_body
        self.num_heads = num_heads
        
        # Wave-to-celestial mapping (dynamic, data-driven)
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
        print(f"   - Input waves: {self.num_input_waves} (auto-detected)")
        print(f"   - Celestial bodies: {len(CelestialBody)}")
        print(f"   - Celestial dimension: {celestial_dim}")
        print(f"   - Dynamic feature mapping: ENABLED")
        print(f"   - Assumes phase info already in inputs (sin/cos encoded)")
        print(f"   - Will compute explicit phase differences for edges")
    
    def _auto_detect_input_waves(self, configured_waves: int) -> int:
        """Auto-detect the correct number of input waves from the CSV file."""
        csv_path = "data/prepared_financial_data.csv"
        try:
            import pandas as pd
            df_header = pd.read_csv(csv_path, nrows=0)
            column_names = df_header.columns.tolist()
            
            # Count actual celestial columns (exclude date, OHLC, time_delta)
            skip_prefixes = ['date', 'log_', 'time_delta']
            celestial_columns = [col for col in column_names 
                               if not any(col.startswith(prefix) for prefix in skip_prefixes)]
            actual_celestial_features = len(celestial_columns)
            
            if configured_waves != actual_celestial_features:
                print(f"🔧 AUTO-CORRECTING INPUT WAVE COUNT:")
                print(f"   Configured: {configured_waves}")
                print(f"   Detected from CSV: {actual_celestial_features}")
                print(f"   Using detected value: {actual_celestial_features}")
                return actual_celestial_features
            else:
                print(f"✅ Input wave count matches CSV: {configured_waves}")
                return configured_waves
                
        except Exception as e:
            print(f"⚠️  Could not auto-detect input waves: {e}")
            print(f"   Using configured value: {configured_waves}")
            return configured_waves

    
    def _create_astrological_mapping(self) -> Dict[CelestialBody, List[int]]:
        """
        FIXED: Dynamic mapping based on actual CSV header structure.
        
        Automatically parses the data file header to create correct celestial body mappings.
        No hardcoding - adapts to any CSV structure with celestial body features.
        
        CRITICAL FIX: Resolves Issue #3 - Feature Mapping Algorithmic Error
        """
        # Load and parse the actual CSV header
        csv_path = "data/prepared_financial_data.csv"
        try:
            import pandas as pd
            df_header = pd.read_csv(csv_path, nrows=0)  # Read only header
            column_names = df_header.columns.tolist()
        except Exception as e:
            print(f"⚠️  Could not read CSV header from {csv_path}: {e}")
            print("⚠️  Falling back to manual column analysis...")
            # Fallback: manual header based on known structure
            column_names = [
                'date', 'log_Open', 'log_High', 'log_Low', 'log_Close', 'time_delta',
                'dyn_Sun_sin', 'dyn_Sun_cos', 'dyn_Sun_speed', 'dyn_Sun_sign_sin', 'dyn_Sun_sign_cos', 'dyn_Sun_distance', 'dyn_Sun_lat',
                'dyn_Moon_sin', 'dyn_Moon_cos', 'dyn_Moon_speed', 'dyn_Moon_sign_sin', 'dyn_Moon_sign_cos', 'dyn_Moon_distance', 'dyn_Moon_lat',
                'dyn_Mars_sin', 'dyn_Mars_cos', 'dyn_Mars_speed', 'dyn_Mars_sign_sin', 'dyn_Mars_sign_cos', 'dyn_Mars_distance', 'dyn_Mars_lat',
                'dyn_Mercury_sin', 'dyn_Mercury_cos', 'dyn_Mercury_speed', 'dyn_Mercury_sign_sin', 'dyn_Mercury_sign_cos', 'dyn_Mercury_distance', 'dyn_Mercury_lat',
                'dyn_Jupiter_sin', 'dyn_Jupiter_cos', 'dyn_Jupiter_speed', 'dyn_Jupiter_sign_sin', 'dyn_Jupiter_sign_cos', 'dyn_Jupiter_distance', 'dyn_Jupiter_lat',
                'dyn_Venus_sin', 'dyn_Venus_cos', 'dyn_Venus_speed', 'dyn_Venus_sign_sin', 'dyn_Venus_sign_cos', 'dyn_Venus_distance', 'dyn_Venus_lat',
                'dyn_Saturn_sin', 'dyn_Saturn_cos', 'dyn_Saturn_speed', 'dyn_Saturn_sign_sin', 'dyn_Saturn_sign_cos', 'dyn_Saturn_distance', 'dyn_Saturn_lat',
                'dyn_Uranus_sin', 'dyn_Uranus_cos', 'dyn_Uranus_speed', 'dyn_Uranus_sign_sin', 'dyn_Uranus_sign_cos', 'dyn_Uranus_distance', 'dyn_Uranus_lat',
                'dyn_Neptune_sin', 'dyn_Neptune_cos', 'dyn_Neptune_speed', 'dyn_Neptune_sign_sin', 'dyn_Neptune_sign_cos', 'dyn_Neptune_distance', 'dyn_Neptune_lat',
                'dyn_Pluto_sin', 'dyn_Pluto_cos', 'dyn_Pluto_speed', 'dyn_Pluto_sign_sin', 'dyn_Pluto_sign_cos', 'dyn_Pluto_distance', 'dyn_Pluto_lat',
                'dyn_Mean Rahu_sin', 'dyn_Mean Rahu_cos', 'dyn_Mean Rahu_speed', 'dyn_Mean Rahu_sign_sin', 'dyn_Mean Rahu_sign_cos',
                'dyn_Mean Ketu_sin', 'dyn_Mean Ketu_cos', 'dyn_Mean Ketu_speed', 'dyn_Mean Ketu_sign_sin', 'dyn_Mean Ketu_sign_cos',
                'dyn_Sun_shadbala', 'dyn_Moon_shadbala', 'dyn_Mars_shadbala', 'dyn_Mercury_shadbala', 'dyn_Jupiter_shadbala', 'dyn_Venus_shadbala', 'dyn_Saturn_shadbala',
                'Sun_sin', 'Sun_cos', 'Moon_sin', 'Moon_cos', 'Mars_sin', 'Mars_cos', 'Mercury_sin', 'Mercury_cos',
                'Jupiter_sin', 'Jupiter_cos', 'Venus_sin', 'Venus_cos', 'Saturn_sin', 'Saturn_cos',
                'Uranus_sin', 'Uranus_cos', 'Neptune_sin', 'Neptune_cos', 'Pluto_sin', 'Pluto_cos',
                'Mean Rahu_sin', 'Mean Rahu_cos', 'Mean Ketu_sin', 'Mean Ketu_cos', 'Ascendant_sin', 'Ascendant_cos'
            ]
        
        print(f"📊 Analyzing CSV with {len(column_names)} columns...")
        
        # Create celestial body name mapping (handle variations in naming)
        celestial_name_mapping = {
            'Sun': CelestialBody.SUN,
            'Moon': CelestialBody.MOON,
            'Mars': CelestialBody.MARS,
            'Mercury': CelestialBody.MERCURY,
            'Jupiter': CelestialBody.JUPITER,
            'Venus': CelestialBody.VENUS,
            'Saturn': CelestialBody.SATURN,
            'Uranus': CelestialBody.URANUS,
            'Neptune': CelestialBody.NEPTUNE,
            'Pluto': CelestialBody.PLUTO,
            'Mean Rahu': CelestialBody.NORTH_NODE,  # Rahu = North Node
            'Mean Ketu': CelestialBody.SOUTH_NODE,  # Ketu = South Node
            'Chiron': CelestialBody.CHIRON,
            # Note: Ascendant will be handled separately for Chiron
        }
        
        # Initialize mapping dictionary
        mapping = {}
        
        # Skip non-celestial columns (OHLC, date, time_delta)
        skip_prefixes = ['date', 'log_', 'time_delta']
        
        # Group features by celestial body
        celestial_features = {}
        used_indices = set()  # Track used indices to avoid overlaps
        
        # First pass: collect all celestial features
        celestial_columns = []
        for idx, col_name in enumerate(column_names):
            # Skip non-celestial columns
            if any(col_name.startswith(prefix) for prefix in skip_prefixes):
                continue
            celestial_columns.append((idx, col_name))
        
        print(f"📊 Found {len(celestial_columns)} celestial columns out of {len(column_names)} total")
        
        # Second pass: assign features to celestial bodies
        unassigned_features = []
        
        for idx, col_name in celestial_columns:
            # Extract celestial body name from column
            celestial_body_name = self._extract_celestial_body_name(col_name)
            
            if celestial_body_name and celestial_body_name in celestial_name_mapping:
                celestial_body = celestial_name_mapping[celestial_body_name]
                
                if celestial_body not in celestial_features:
                    celestial_features[celestial_body] = []
                
                # Only add if not already used (avoid overlaps)
                if idx not in used_indices:
                    celestial_features[celestial_body].append(idx)
                    used_indices.add(idx)
            else:
                # Track unassigned features for debugging
                unassigned_features.append((idx, col_name))
        
        # Report unassigned features for debugging
        if unassigned_features:
            print(f"⚠️  {len(unassigned_features)} features could not be assigned to celestial bodies:")
            for idx, col_name in unassigned_features[:10]:  # Show first 10
                print(f"   {idx}: {col_name}")
            if len(unassigned_features) > 10:
                print(f"   ... and {len(unassigned_features) - 10} more")
        
        # Create final mapping, ensuring all celestial bodies are represented
        for celestial_body in CelestialBody:
            if celestial_body in celestial_features and celestial_features[celestial_body]:
                # Convert to 0-based indexing for the celestial-only features
                original_indices = celestial_features[celestial_body]
                # Map from original CSV indices to celestial feature indices (0-based)
                celestial_indices = []
                for orig_idx in original_indices:
                    # Find position in celestial_columns
                    celestial_pos = next((i for i, (idx, _) in enumerate(celestial_columns) if idx == orig_idx), None)
                    if celestial_pos is not None:
                        celestial_indices.append(celestial_pos)
                mapping[celestial_body] = celestial_indices
            else:
                # If a celestial body is missing, create empty mapping for now
                print(f"⚠️  {celestial_body.value} not found in CSV")
                mapping[celestial_body] = []
        
        # Handle special case: Chiron is not in the CSV, so assign it to Ascendant features
        if not mapping[CelestialBody.CHIRON] and unassigned_features:
            # Find Ascendant features for Chiron
            ascendant_features = [(idx, col) for idx, col in unassigned_features if 'Ascendant' in col]
            if ascendant_features:
                print(f"📊 Assigning Ascendant features to Chiron (closest astrological match)")
                chiron_indices = []
                for orig_idx, col_name in ascendant_features:
                    celestial_pos = next((i for i, (idx, _) in enumerate(celestial_columns) if idx == orig_idx), None)
                    if celestial_pos is not None:
                        chiron_indices.append(celestial_pos)
                        used_indices.add(orig_idx)
                mapping[CelestialBody.CHIRON] = chiron_indices
        
        # Validation and reporting
        self._validate_and_report_mapping(mapping, column_names)
        
        return mapping
    
    def _extract_celestial_body_name(self, column_name: str) -> Optional[str]:
        """
        Extract celestial body name from column name.
        
        Handles various naming patterns:
        - dyn_Sun_sin -> Sun
        - Sun_sin -> Sun  
        - dyn_Mean Rahu_cos -> Mean Rahu
        - dyn_Sun_shadbala -> Sun
        - dyn_Sun_sign_sin -> Sun (not sign!)
        """
        # Remove common prefixes
        clean_name = column_name.replace('dyn_', '').replace('static_', '')
        
        # Handle special cases first (order matters!)
        if 'Mean Rahu' in clean_name:
            return 'Mean Rahu'
        elif 'Mean Ketu' in clean_name:
            return 'Mean Ketu'
        elif 'Ascendant' in clean_name:
            return 'Ascendant'  # Special case for Ascendant
        
        # List of known celestial bodies (in order of specificity)
        celestial_bodies = [
            'Mercury', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto',
            'Venus', 'Mars', 'Moon', 'Sun'  # Put shorter names last to avoid conflicts
        ]
        
        # Find the celestial body name in the column
        for body in celestial_bodies:
            if body in clean_name:
                # Verify it's actually the body name, not part of another word
                # Check if it's followed by underscore or end of string
                body_index = clean_name.find(body)
                if body_index != -1:
                    # Check what comes after the body name
                    after_body = clean_name[body_index + len(body):]
                    if after_body == '' or after_body.startswith('_'):
                        return body
        
        return None
    
    def _validate_and_report_mapping(self, mapping: Dict[CelestialBody, List[int]], column_names: List[str]) -> None:
        """Validate the mapping and report statistics."""
        print(f"\n🌌 CELESTIAL FEATURE MAPPING REPORT:")
        print(f"{'='*60}")
        
        # Get celestial columns only (skip OHLC, date, time_delta)
        skip_prefixes = ['date', 'log_', 'time_delta']
        celestial_columns = [(idx, col) for idx, col in enumerate(column_names) 
                           if not any(col.startswith(prefix) for prefix in skip_prefixes)]
        
        total_features = 0
        max_index = -1
        
        for body, indices in mapping.items():
            # Get actual feature names for these indices
            feature_names = []
            for idx in indices:
                if idx < len(celestial_columns):
                    _, col_name = celestial_columns[idx]
                    feature_names.append(col_name)
                else:
                    feature_names.append(f"INDEX_{idx}")
            
            total_features += len(indices)
            if indices:
                max_index = max(max_index, max(indices))
            
            print(f"🪐 {body.value:12} ({len(indices):2d} features): {indices}")
            if len(feature_names) <= 5:
                print(f"   Features: {', '.join(feature_names)}")
            else:
                print(f"   Features: {', '.join(feature_names[:3])} ... {', '.join(feature_names[-2:])}")
        
        print(f"{'='*60}")
        print(f"📊 Total celestial features mapped: {total_features}")
        print(f"📊 Available celestial columns: {len(celestial_columns)}")
        print(f"📊 Total CSV columns: {len(column_names)}")
        print(f"📊 Expected input waves: {self.num_input_waves}")
        print(f"📊 Max feature index used: {max_index}")
        
        # Validation checks
        expected_celestial_features = len(celestial_columns)
        if total_features != expected_celestial_features:
            print(f"⚠️  WARNING: Feature count mismatch!")
            print(f"   Expected celestial features: {expected_celestial_features}")
            print(f"   Mapped features: {total_features}")
        
        # Check for overlapping indices
        all_indices = []
        for indices in mapping.values():
            all_indices.extend(indices)
        
        if len(all_indices) != len(set(all_indices)):
            print(f"⚠️  WARNING: Overlapping feature indices detected!")
        
        # Check index bounds
        if max_index >= len(celestial_columns):
            print(f"⚠️  WARNING: Index out of bounds! Max index: {max_index}, Available: {len(celestial_columns)}")
        
        print(f"✅ Celestial mapping validation complete.\n")
    
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
        
        # Extract speed/velocity - 3rd feature if available
        if num_features >= 3:
            phase_info['speed'] = features[:, :, 2]
        else:
            # For bodies with fewer features, use a default speed
            phase_info['speed'] = torch.zeros_like(features[:, :, 0])
        
        # Extract zodiac sign phase (φ) - 4th and 5th features if available
        if num_features >= 5:
            sin_sign = features[:, :, 3]
            cos_sign = features[:, :, 4]
            sign_phase = torch.atan2(sin_sign, cos_sign)
            
            phase_info['sign_phase'] = sign_phase  # φ phase
            phase_info['sign_sin'] = sin_sign
            phase_info['sign_cos'] = cos_sign
        else:
            # For bodies with fewer features, use longitude phase as sign phase
            phase_info['sign_phase'] = longitude_phase
            phase_info['sign_sin'] = sin_longitude
            phase_info['sign_cos'] = cos_longitude
        
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
            # For bodies with fewer features, use defaults
            phase_info['distance'] = torch.ones_like(features[:, :, 0])  # Default distance
            phase_info['latitude'] = torch.zeros_like(features[:, :, 0])  # Default latitude
            static_start_idx = min(5, num_features)
            if self.has_shadbala and num_features > static_start_idx:
                phase_info['shadbala'] = features[:, :, static_start_idx]
                static_start_idx += 1
        
        # Extract ecliptic longitude (fundamental astrological coordinate - 2D projection)
        if num_features >= static_start_idx + 2:
            sin_ecliptic = features[:, :, static_start_idx]
            cos_ecliptic = features[:, :, static_start_idx + 1]
            ecliptic_longitude = torch.atan2(sin_ecliptic, cos_ecliptic)
            
            phase_info['ecliptic_longitude'] = ecliptic_longitude  # Fundamental astrological angle
            phase_info['ecliptic_sin'] = sin_ecliptic
            phase_info['ecliptic_cos'] = cos_ecliptic
        else:
            # For bodies with fewer features, use longitude phase as ecliptic
            phase_info['ecliptic_longitude'] = longitude_phase
            phase_info['ecliptic_sin'] = sin_longitude
            phase_info['ecliptic_cos'] = cos_longitude
        
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
                'theta': safe_extract_last(body_info.get('longitude_phase'), (batch_size,)),  # FIXED: longitude_phase -> theta
                'phi': safe_extract_last(body_info.get('sign_phase'), (batch_size,)),         # FIXED: sign_phase -> phi
                'velocity': safe_extract_last(body_info.get('speed'), (batch_size,)),         # FIXED: speed -> velocity
                'radius': safe_extract_last(body_info.get('distance'), (batch_size,)),        # FIXED: distance -> radius
                'longitude': safe_extract_last(body_info.get('ecliptic_longitude'), (batch_size,))  # FIXED: ecliptic_longitude -> longitude
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
    
    def forward_rich_features(
        self, 
        celestial_tensor: torch.Tensor,
        phase_info: Dict
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute rich edge feature VECTORS (NO compression to scalars!)
        
        This is the Petri net version that preserves ALL edge information.
        
        Args:
            celestial_tensor: [batch, seq_len, num_bodies, celestial_dim]
            phase_info: Dictionary with phase information
            
        Returns:
            Tuple of:
                - edge_features: [batch, seq_len, num_bodies, num_bodies, 6] 
                  Full feature vectors with [theta_diff, phi_diff, velocity_diff, 
                  radius_ratio, longitude_diff, phase_alignment]
                - metadata: Diagnostics
        """
        batch_size, seq_len, num_bodies, celestial_dim = celestial_tensor.shape
        device = celestial_tensor.device
        
        # Initialize edge feature tensor (VECTORS, not scalars!)
        edge_features = torch.zeros(
            batch_size, seq_len, num_bodies, num_bodies, 6,
            device=device, dtype=torch.float32
        )
        
        # Extract phase information for all timesteps
        body_names = list(CelestialBody)
        all_timestep_phases = {}
        
        for i, body in enumerate(body_names):
            body_info = phase_info[body.value]
            
            all_timestep_phases[i] = {
                'theta': self._safe_extract_all_timesteps(
                    body_info.get('longitude_phase'), batch_size, seq_len, device
                ),
                'phi': self._safe_extract_all_timesteps(
                    body_info.get('sign_phase'), batch_size, seq_len, device
                ),
                'velocity': self._safe_extract_all_timesteps(
                    body_info.get('speed'), batch_size, seq_len, device
                ),
                'radius': self._safe_extract_all_timesteps(
                    body_info.get('distance'), batch_size, seq_len, device
                ),
                'longitude': self._safe_extract_all_timesteps(
                    body_info.get('ecliptic_longitude'), batch_size, seq_len, device
                )
            }
        
        # Compute edge features for all pairs (preserve full temporal dimension)
        phase_diff_stats = {
            'theta_diffs': [],
            'phi_diffs': [],
            'velocity_ratios': [],
            'radius_ratios': [],
            'longitude_diffs': [],
            'phase_alignments': []
        }
        
        for i in range(num_bodies):
            for j in range(num_bodies):  # All edges including self-loops
                # Compute phase differences across all timesteps
                theta_diff = self._circular_difference(
                    all_timestep_phases[i]['theta'],
                    all_timestep_phases[j]['theta']
                )  # [batch, seq_len]
                
                phi_diff = self._circular_difference(
                    all_timestep_phases[i]['phi'],
                    all_timestep_phases[j]['phi']
                )  # [batch, seq_len]
                
                velocity_ratio = (
                    all_timestep_phases[i]['velocity'] / 
                    (all_timestep_phases[j]['velocity'] + 1e-8)
                )  # [batch, seq_len]
                
                radius_ratio = (
                    all_timestep_phases[i]['radius'] / 
                    (all_timestep_phases[j]['radius'] + 1e-8)
                )  # [batch, seq_len]
                
                longitude_diff = self._circular_difference(
                    all_timestep_phases[i]['longitude'],
                    all_timestep_phases[j]['longitude']
                )  # [batch, seq_len]
                
                # Phase alignment measure
                phase_alignment = (
                    torch.cos(theta_diff) * torch.cos(phi_diff)
                )  # [batch, seq_len]
                
                # --- STABILIZATION: Squash unbounded ratio features to [-1, 1] range ---
                # Prevents exploding gradients from outlier values (e.g., extreme radius/velocity ratios)
                stabilized_velocity_ratio = torch.tanh(velocity_ratio)
                stabilized_radius_ratio = torch.tanh(radius_ratio)
                # --- END STABILIZATION ---
                
                # Stack all features (PRESERVED AS VECTORS!)
                edge_features[:, :, i, j, 0] = theta_diff  # Already bounded by circular_difference [-π, π]
                edge_features[:, :, i, j, 1] = phi_diff    # Already bounded by circular_difference [-π, π]
                edge_features[:, :, i, j, 2] = stabilized_velocity_ratio  # Stabilized to [-1, 1]
                edge_features[:, :, i, j, 3] = stabilized_radius_ratio   # Stabilized to [-1, 1]
                edge_features[:, :, i, j, 4] = longitude_diff  # Already bounded by circular_difference [-π, π]
                edge_features[:, :, i, j, 5] = phase_alignment # Already bounded by cosine products [-1, 1]
                
                # Collect statistics (only for i < j to avoid duplicates)
                if i < j:
                    # Store raw values for statistics (before stabilization)
                    phase_diff_stats['theta_diffs'].append(theta_diff.abs().mean().item())
                    phase_diff_stats['phi_diffs'].append(phi_diff.abs().mean().item())
                    phase_diff_stats['velocity_ratios'].append(velocity_ratio.mean().item())  # Raw ratio value
                    phase_diff_stats['radius_ratios'].append(radius_ratio.mean().item())  # Raw ratio value
                    phase_diff_stats['longitude_diffs'].append(longitude_diff.abs().mean().item())
                    phase_diff_stats['phase_alignments'].append(phase_alignment.mean().item())
        
        # Compute metadata
        metadata = {
            'avg_theta_diff': sum(phase_diff_stats['theta_diffs']) / len(phase_diff_stats['theta_diffs']),
            'avg_phi_diff': sum(phase_diff_stats['phi_diffs']) / len(phase_diff_stats['phi_diffs']),
            'avg_velocity_ratio': sum(phase_diff_stats['velocity_ratios']) / len(phase_diff_stats['velocity_ratios']),
            'avg_radius_ratio': sum(phase_diff_stats['radius_ratios']) / len(phase_diff_stats['radius_ratios']),
            'avg_longitude_diff': sum(phase_diff_stats['longitude_diffs']) / len(phase_diff_stats['longitude_diffs']),
            'avg_phase_alignment': sum(phase_diff_stats['phase_alignments']) / len(phase_diff_stats['phase_alignments']),
            'edge_feature_dim': 6,
            'edge_features_preserved': True,
            'no_compression': True,
            'stabilization_applied': True,  # NEW: Indicates tanh squashing is active
            'stabilized_features': ['velocity_ratio', 'radius_ratio']  # NEW: Which features are stabilized
        }
        
        return edge_features, metadata
    
    def _safe_extract_all_timesteps(
        self,
        tensor: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Extract all timesteps from phase information tensor."""
        if tensor is None:
            return torch.zeros(batch_size, seq_len, device=device)
        
        if tensor.dim() == 1:
            # Broadcast across seq_len
            return tensor.unsqueeze(-1).expand(-1, seq_len)
        elif tensor.dim() == 2:
            # Already [batch, seq_len]
            return tensor
        else:
            # Reshape to [batch, seq_len]
            return tensor.reshape(batch_size, seq_len)
    
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