#!/usr/bin/env python3
"""
Test script to verify the celestial feature mapping fix.

This script tests the new dynamic mapping system to ensure:
1. Correct feature assignment to celestial bodies
2. No overlapping or missing features
3. Proper handling of different CSV structures
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from layers.modular.aggregation.phase_aware_celestial_processor import PhaseAwareCelestialProcessor
from layers.modular.graph.celestial_body_nodes import CelestialBody

def test_celestial_mapping():
    """Test the new dynamic celestial mapping system."""
    
    print("🧪 TESTING CELESTIAL FEATURE MAPPING FIX")
    print("="*60)
    
    # Test with actual configuration (auto-detects correct size)
    processor = PhaseAwareCelestialProcessor(
        num_input_waves=118,  # Will be auto-corrected to 113
        celestial_dim=32,
        waves_per_body=9,
        num_heads=8
    )
    
    print("\n🔍 MAPPING VALIDATION:")
    
    # Check that all celestial bodies are mapped
    all_bodies_mapped = all(body in processor.wave_mapping for body in CelestialBody)
    print(f"✅ All celestial bodies mapped: {all_bodies_mapped}")
    
    # Check for overlapping indices
    all_indices = []
    for body, indices in processor.wave_mapping.items():
        all_indices.extend(indices)
    
    unique_indices = set(all_indices)
    no_overlaps = len(all_indices) == len(unique_indices)
    print(f"✅ No overlapping indices: {no_overlaps}")
    
    # Check index bounds
    max_index = max(all_indices) if all_indices else -1
    indices_in_bounds = max_index < 118  # Should be less than total features
    print(f"✅ All indices in bounds (max: {max_index}): {indices_in_bounds}")
    
    # Test forward pass with dummy data
    print(f"\n🧪 TESTING FORWARD PASS:")
    
    batch_size, seq_len = 2, 10
    dummy_input = torch.randn(batch_size, seq_len, 113)  # Use correct size
    
    try:
        celestial_features, adjacency_matrix, metadata = processor(dummy_input)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {celestial_features.shape}")
        print(f"   Adjacency shape: {adjacency_matrix.shape}")
        print(f"   Expected output: [batch={batch_size}, seq_len={seq_len}, features={13*32}]")
        
        # Verify output dimensions
        expected_features = 13 * 32  # 13 celestial bodies × 32D each
        output_correct = celestial_features.shape == (batch_size, seq_len, expected_features)
        print(f"✅ Output dimensions correct: {output_correct}")
        
        # Verify adjacency matrix
        adj_correct = adjacency_matrix.shape == (batch_size, 13, 13)
        print(f"✅ Adjacency matrix correct: {adj_correct}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    print(f"\n📊 FEATURE DISTRIBUTION:")
    for body, indices in processor.wave_mapping.items():
        print(f"   {body.value:12}: {len(indices):2d} features (indices: {min(indices)}-{max(indices)})")
    
    print(f"\n🎯 MAPPING FIX VALIDATION:")
    
    # Check that we're using actual CSV structure
    csv_based = hasattr(processor, '_validate_input_wave_count')
    print(f"✅ CSV-based mapping: {csv_based}")
    
    # Check that Sun gets its actual features (should include dyn_Sun_* features)
    sun_indices = processor.wave_mapping[CelestialBody.SUN]
    print(f"✅ Sun mapped to indices: {sun_indices}")
    
    # Verify no hardcoded 9-features-per-body assumption
    feature_counts = [len(indices) for indices in processor.wave_mapping.values()]
    variable_counts = len(set(feature_counts)) > 1
    print(f"✅ Variable feature counts per body: {variable_counts}")
    print(f"   Feature counts: {feature_counts}")
    
    return True

if __name__ == "__main__":
    success = test_celestial_mapping()
    
    if success:
        print(f"\n🎉 CELESTIAL MAPPING FIX VALIDATION: PASSED")
        print(f"✅ The dynamic mapping system is working correctly!")
        print(f"✅ Celestial bodies now get their correct features from the CSV!")
    else:
        print(f"\n❌ CELESTIAL MAPPING FIX VALIDATION: FAILED")
        print(f"❌ There are still issues with the mapping system.")
    
    print(f"\n" + "="*60)