"""
Test Phase-Aware Celestial Processor

This module tests the phase-aware celestial processor that handles the actual
input feature structure with sin/cos encoded phases, velocities, and distances.
"""

import torch
import pytest
import numpy as np
from layers.modular.aggregation.phase_aware_celestial_processor import (
    PhaseAwareCelestialProcessor,
    PhaseExtractor,
    PhaseDifferenceEdgeComputer,
    PhaseAwareAttention
)
from layers.modular.graph.celestial_body_nodes import CelestialBody


class TestPhaseAwareCelestialProcessor:
    """Test the main phase-aware celestial processor."""
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input data matching the actual feature structure."""
        batch_size, seq_len, num_features = 4, 96, 118
        
        # Create realistic celestial data
        torch.manual_seed(42)
        wave_features = torch.randn(batch_size, seq_len, num_features)
        
        # Make some features look like sin/cos pairs (bounded to [-1, 1])
        for i in range(0, num_features, 2):
            if i + 1 < num_features:
                # Create sin/cos pairs that are actually on the unit circle
                angles = torch.rand(batch_size, seq_len) * 2 * np.pi
                wave_features[:, :, i] = torch.sin(angles)  # sin
                wave_features[:, :, i + 1] = torch.cos(angles)  # cos
        
        return wave_features
    
    @pytest.fixture
    def processor(self):
        """Create a phase-aware celestial processor."""
        return PhaseAwareCelestialProcessor(
            num_input_waves=118,
            celestial_dim=32,
            waves_per_body=9,
            num_heads=4
        )
    
    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.num_input_waves == 118
        assert processor.celestial_dim == 32
        assert processor.num_heads == 4
        assert len(processor.wave_mapping) > 0
        
        # Check that celestial processors are created
        assert len(processor.celestial_processors) > 0
        
        # Check that phase difference computer exists
        assert processor.phase_difference_computer is not None
        
        # Check that inter-celestial attention exists
        assert processor.inter_celestial_attention is not None
    
    def test_wave_mapping(self, processor):
        """Test that wave mapping is correctly structured."""
        mapping = processor.wave_mapping
        
        # Check that we have mappings for expected celestial bodies
        expected_bodies = [
            CelestialBody.SUN, CelestialBody.MOON, CelestialBody.MARS,
            CelestialBody.MERCURY, CelestialBody.JUPITER, CelestialBody.VENUS,
            CelestialBody.SATURN, CelestialBody.URANUS, CelestialBody.NEPTUNE,
            CelestialBody.PLUTO, CelestialBody.NORTH_NODE, CelestialBody.SOUTH_NODE
        ]
        
        for body in expected_bodies:
            assert body in mapping, f"Missing mapping for {body}"
            assert len(mapping[body]) > 0, f"Empty mapping for {body}"
        
        # Check that indices are within valid range
        all_indices = []
        for indices in mapping.values():
            all_indices.extend(indices)
        
        assert min(all_indices) >= 5, "Indices should start after OHLC + time_delta"
        assert max(all_indices) < 118, "Indices should be within input range"
    
    def test_forward_pass(self, processor, sample_input):
        """Test forward pass with sample input."""
        celestial_features, adjacency_matrix, metadata = processor(sample_input)
        
        batch_size, seq_len = sample_input.shape[:2]
        expected_celestial_dim = len(processor.wave_mapping) * processor.celestial_dim
        
        # Check output shapes
        assert celestial_features.shape == (batch_size, seq_len, expected_celestial_dim)
        assert adjacency_matrix.shape == (batch_size, len(processor.wave_mapping), len(processor.wave_mapping))
        
        # Check that adjacency matrix values are in valid range [0, 1]
        assert torch.all(adjacency_matrix >= 0), "Adjacency values should be non-negative"
        assert torch.all(adjacency_matrix <= 1), "Adjacency values should be <= 1"
        
        # Check that adjacency matrix is symmetric
        assert torch.allclose(adjacency_matrix, adjacency_matrix.transpose(-2, -1), atol=1e-6), \
            "Adjacency matrix should be symmetric"
        
        # Check metadata structure
        assert 'celestial_metadata' in metadata
        assert 'phase_info' in metadata
        assert 'edge_metadata' in metadata
        assert 'phase_coherence' in metadata
    
    def test_phase_extraction(self, processor, sample_input):
        """Test that phase information is correctly extracted."""
        _, _, metadata = processor(sample_input)
        
        phase_info = metadata['phase_info']
        
        # Check that we have phase info for each celestial body
        for body in processor.wave_mapping.keys():
            assert body.value in phase_info, f"Missing phase info for {body.value}"
            
            body_phase_info = phase_info[body.value]
            
            # Check that basic phase information is present
            expected_keys = ['longitude_phase', 'longitude_sin', 'longitude_cos', 'speed']
            for key in expected_keys:
                if key in body_phase_info:  # Some bodies might not have all features
                    phase_data = body_phase_info[key]
                    assert isinstance(phase_data, torch.Tensor), f"{key} should be a tensor"
                    assert phase_data.shape[:2] == sample_input.shape[:2], f"Wrong shape for {key}"
    
    def test_edge_computation(self, processor, sample_input):
        """Test that edges are computed based on phase differences."""
        _, adjacency_matrix, metadata = processor(sample_input)
        
        edge_metadata = metadata['edge_metadata']
        
        # Check edge metadata
        expected_edge_keys = [
            'avg_theta_diff', 'avg_phi_diff', 'avg_velocity_diff',
            'avg_edge_strength', 'edge_density'
        ]
        
        for key in expected_edge_keys:
            if key in edge_metadata:  # Some keys might be missing if features aren't available
                assert isinstance(edge_metadata[key], (int, float)), f"{key} should be numeric"
                assert edge_metadata[key] >= 0, f"{key} should be non-negative"
        
        # Check that edge density is reasonable
        if 'edge_density' in edge_metadata:
            assert 0 <= edge_metadata['edge_density'] <= 1, "Edge density should be between 0 and 1"
    
    def test_phase_coherence(self, processor, sample_input):
        """Test phase coherence computation."""
        _, _, metadata = processor(sample_input)
        
        phase_coherence = metadata['phase_coherence']
        
        # Check coherence values
        for coherence_type, value in phase_coherence.items():
            assert isinstance(value, (int, float)), f"{coherence_type} should be numeric"
            assert 0 <= value <= 1, f"{coherence_type} should be between 0 and 1"


class TestPhaseExtractor:
    """Test the phase extractor component."""
    
    @pytest.fixture
    def sample_celestial_features(self):
        """Create sample features for a celestial body."""
        batch_size, seq_len = 4, 96
        
        # Create features matching the expected structure:
        # [sin_longitude, cos_longitude, speed, sign_sin, sign_cos, distance, lat, shadbala, ecliptic_sin, ecliptic_cos]
        num_features = 10
        features = torch.randn(batch_size, seq_len, num_features)
        
        # Make sin/cos pairs realistic
        angles1 = torch.rand(batch_size, seq_len) * 2 * np.pi
        features[:, :, 0] = torch.sin(angles1)  # sin_longitude
        features[:, :, 1] = torch.cos(angles1)  # cos_longitude
        
        angles2 = torch.rand(batch_size, seq_len) * 2 * np.pi
        features[:, :, 3] = torch.sin(angles2)  # sign_sin
        features[:, :, 4] = torch.cos(angles2)  # sign_cos
        
        angles3 = torch.rand(batch_size, seq_len) * 2 * np.pi
        features[:, :, 8] = torch.sin(angles3)  # ecliptic_sin
        features[:, :, 9] = torch.cos(angles3)  # ecliptic_cos
        
        return features
    
    def test_phase_extraction_sun(self, sample_celestial_features):
        """Test phase extraction for Sun (has all features)."""
        extractor = PhaseExtractor('Sun')
        phase_info = extractor(sample_celestial_features)
        
        # Check that all expected phases are extracted
        expected_keys = [
            'longitude_phase', 'longitude_sin', 'longitude_cos',
            'speed', 'sign_phase', 'sign_sin', 'sign_cos',
            'distance', 'latitude', 'shadbala',
            'ecliptic_longitude', 'ecliptic_sin', 'ecliptic_cos',
            'phase_velocity'
        ]
        
        for key in expected_keys:
            assert key in phase_info, f"Missing {key} in phase info"
            assert isinstance(phase_info[key], torch.Tensor), f"{key} should be a tensor"
    
    def test_phase_extraction_uranus(self, sample_celestial_features):
        """Test phase extraction for Uranus (no shadbala)."""
        # Remove shadbala feature
        features_no_shadbala = sample_celestial_features[:, :, [0, 1, 2, 3, 4, 5, 6, 8, 9]]
        
        extractor = PhaseExtractor('Uranus')
        phase_info = extractor(features_no_shadbala)
        
        # Should have most features but no shadbala
        assert 'longitude_phase' in phase_info
        assert 'speed' in phase_info
        assert 'distance' in phase_info
        assert 'ecliptic_longitude' in phase_info
        assert 'shadbala' not in phase_info  # Should not be present
    
    def test_phase_extraction_north_node(self):
        """Test phase extraction for North Node (limited features)."""
        batch_size, seq_len = 4, 96
        
        # North Node has only: [sin_longitude, cos_longitude, speed, sign_sin, sign_cos, ecliptic_sin, ecliptic_cos]
        num_features = 7
        features = torch.randn(batch_size, seq_len, num_features)
        
        # Make sin/cos pairs realistic
        angles1 = torch.rand(batch_size, seq_len) * 2 * np.pi
        features[:, :, 0] = torch.sin(angles1)
        features[:, :, 1] = torch.cos(angles1)
        
        angles2 = torch.rand(batch_size, seq_len) * 2 * np.pi
        features[:, :, 3] = torch.sin(angles2)
        features[:, :, 4] = torch.cos(angles2)
        
        angles3 = torch.rand(batch_size, seq_len) * 2 * np.pi
        features[:, :, 5] = torch.sin(angles3)
        features[:, :, 6] = torch.cos(angles3)
        
        extractor = PhaseExtractor('North_Node')
        phase_info = extractor(features)
        
        # Should have basic features but no distance/latitude/shadbala
        assert 'longitude_phase' in phase_info
        assert 'speed' in phase_info
        assert 'sign_phase' in phase_info
        assert 'ecliptic_longitude' in phase_info
        assert 'distance' not in phase_info
        assert 'latitude' not in phase_info
        assert 'shadbala' not in phase_info
    
    def test_phase_velocity_computation(self, sample_celestial_features):
        """Test that phase velocity is correctly computed."""
        extractor = PhaseExtractor('Sun')
        phase_info = extractor(sample_celestial_features)
        
        phase_velocity = phase_info['phase_velocity']
        longitude_phase = phase_info['longitude_phase']
        
        # Check shapes
        assert phase_velocity.shape == longitude_phase.shape
        
        # First timestep should be zero (no previous timestep)
        assert torch.allclose(phase_velocity[:, 0], torch.zeros_like(phase_velocity[:, 0]))
        
        # Check that velocity is bounded (should be reasonable for phase differences)
        assert torch.all(torch.abs(phase_velocity) <= np.pi), "Phase velocity should be bounded by π"


class TestPhaseDifferenceEdgeComputer:
    """Test the phase difference edge computer."""
    
    @pytest.fixture
    def sample_celestial_tensor(self):
        """Create sample celestial tensor."""
        batch_size, seq_len, num_bodies, celestial_dim = 4, 96, 12, 32
        return torch.randn(batch_size, seq_len, num_bodies, celestial_dim)
    
    @pytest.fixture
    def sample_phase_info(self):
        """Create sample phase information."""
        batch_size, seq_len = 4, 96
        
        phase_info = {}
        body_names = ['sun', 'moon', 'mars', 'mercury', 'jupiter', 'venus', 
                     'saturn', 'uranus', 'neptune', 'pluto', 'north_node', 'south_node']
        
        for body in body_names:
            # Create realistic phase data
            longitude_phase = torch.rand(batch_size, seq_len) * 2 * np.pi - np.pi
            sign_phase = torch.rand(batch_size, seq_len) * 2 * np.pi - np.pi
            velocity = torch.randn(batch_size, seq_len) * 0.1  # Small velocities
            
            phase_info[body] = {
                'longitude_phase': longitude_phase,
                'sign_phase': sign_phase,
                'velocity': velocity,
                'distance': torch.rand(batch_size, seq_len) + 0.5,  # Positive distances
                'longitude': torch.rand(batch_size, seq_len) * 2 * np.pi - np.pi
            }
        
        return phase_info
    
    def test_edge_computation(self, sample_celestial_tensor, sample_phase_info):
        """Test edge computation from phase differences."""
        edge_computer = PhaseDifferenceEdgeComputer(
            celestial_dim=32,
            num_celestial_bodies=12
        )
        
        adjacency_matrix, metadata = edge_computer(sample_celestial_tensor, sample_phase_info)
        
        batch_size, num_bodies = sample_celestial_tensor.shape[0], sample_celestial_tensor.shape[2]
        
        # Check output shape
        assert adjacency_matrix.shape == (batch_size, num_bodies, num_bodies)
        
        # Check that adjacency matrix is symmetric
        assert torch.allclose(adjacency_matrix, adjacency_matrix.transpose(-2, -1), atol=1e-6)
        
        # Check that diagonal is zero (no self-connections)
        diagonal = torch.diagonal(adjacency_matrix, dim1=-2, dim2=-1)
        assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6)
        
        # Check that values are in [0, 1] range
        assert torch.all(adjacency_matrix >= 0)
        assert torch.all(adjacency_matrix <= 1)
        
        # Check metadata
        assert 'avg_edge_strength' in metadata
        assert 'edge_density' in metadata
        assert isinstance(metadata['avg_edge_strength'], (int, float))
        assert isinstance(metadata['edge_density'], (int, float))


if __name__ == "__main__":
    # Run basic tests
    processor = PhaseAwareCelestialProcessor(
        num_input_waves=118,
        celestial_dim=32,
        waves_per_body=9,
        num_heads=4
    )
    
    # Create sample input
    batch_size, seq_len, num_features = 2, 24, 118
    sample_input = torch.randn(batch_size, seq_len, num_features)
    
    # Make some features look like sin/cos pairs
    for i in range(0, num_features, 2):
        if i + 1 < num_features:
            angles = torch.rand(batch_size, seq_len) * 2 * np.pi
            sample_input[:, :, i] = torch.sin(angles)
            sample_input[:, :, i + 1] = torch.cos(angles)
    
    print("🧪 Testing Phase-Aware Celestial Processor...")
    
    try:
        celestial_features, adjacency_matrix, metadata = processor(sample_input)
        print(f"✅ Forward pass successful!")
        print(f"   - Input shape: {sample_input.shape}")
        print(f"   - Celestial features shape: {celestial_features.shape}")
        print(f"   - Adjacency matrix shape: {adjacency_matrix.shape}")
        print(f"   - Edge density: {metadata['edge_metadata'].get('avg_edge_strength', 'N/A'):.4f}")
        print(f"   - Phase coherence: {metadata['phase_coherence']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("🧪 Phase-Aware Processor tests completed!")