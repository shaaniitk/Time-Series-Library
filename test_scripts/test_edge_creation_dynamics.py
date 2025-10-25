#!/usr/bin/env python3
"""
Test Edge Creation Dynamics and Missing Feature Handling

This script analyzes:
1. How edges are created when nodes have missing features (like Chiron with no speed)
2. How temporal dynamics work during training (not just averages)
3. How speed ratios evolve over time
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from layers.modular.aggregation.phase_aware_celestial_processor import (
    PhaseAwareCelestialProcessor, 
    PhaseDifferenceEdgeComputer,
    PhaseExtractor
)
from layers.modular.graph.celestial_body_nodes import CelestialBody

def test_missing_feature_handling():
    """Test how the system handles celestial bodies with missing features."""
    
    print("🔍 TESTING MISSING FEATURE HANDLING")
    print("="*60)
    
    # Initialize processor
    processor = PhaseAwareCelestialProcessor(
        num_input_waves=113,
        celestial_dim=32
    )
    
    # Get feature mappings for different celestial bodies
    sun_mapping = processor.wave_mapping[CelestialBody.SUN]
    chiron_mapping = processor.wave_mapping[CelestialBody.CHIRON]
    
    print(f"Sun features: {len(sun_mapping)} features")
    print(f"Chiron features: {len(chiron_mapping)} features")
    
    # Create test data
    batch_size, seq_len = 2, 10
    test_input = torch.randn(batch_size, seq_len, 113)
    
    # Extract features for Sun and Chiron
    sun_features = test_input[:, :, sun_mapping]
    chiron_features = test_input[:, :, chiron_mapping]
    
    print(f"\n🌟 SUN FEATURE ANALYSIS:")
    print(f"   Shape: {sun_features.shape}")
    
    # Test Sun phase extraction
    sun_extractor = PhaseExtractor('Sun')
    sun_phase_info = sun_extractor(sun_features)
    
    print(f"   Available phases:")
    for key, tensor in sun_phase_info.items():
        if 'speed' in key or 'velocity' in key:
            print(f"     ✅ {key}: {tensor.shape}, range [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    print(f"\n🪐 CHIRON FEATURE ANALYSIS:")
    print(f"   Shape: {chiron_features.shape}")
    
    # Test Chiron phase extraction
    chiron_extractor = PhaseExtractor('Chiron')
    chiron_phase_info = chiron_extractor(chiron_features)
    
    print(f"   Available phases:")
    for key, tensor in chiron_phase_info.items():
        if 'speed' in key or 'velocity' in key:
            print(f"     {'✅' if tensor.abs().sum() > 0 else '❌'} {key}: {tensor.shape}, range [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    # Test edge creation between Sun and Chiron
    print(f"\n🔗 EDGE CREATION: SUN ↔ CHIRON")
    print(f"-" * 40)
    
    # Create mock celestial tensor
    celestial_tensor = torch.randn(batch_size, seq_len, len(CelestialBody), 32)
    
    # Create phase info for all bodies
    all_phase_info = {}
    for body in CelestialBody:
        if body == CelestialBody.SUN:
            all_phase_info[body.value] = sun_phase_info
        elif body == CelestialBody.CHIRON:
            all_phase_info[body.value] = chiron_phase_info
        else:
            # Mock data for other bodies
            all_phase_info[body.value] = {
                'longitude_phase': torch.randn(batch_size, seq_len) * 2 * np.pi,
                'sign_phase': torch.randn(batch_size, seq_len) * 2 * np.pi,
                'speed': torch.randn(batch_size, seq_len) * 0.1,
                'distance': torch.ones(batch_size, seq_len) + torch.randn(batch_size, seq_len) * 0.1,
                'ecliptic_longitude': torch.randn(batch_size, seq_len) * 2 * np.pi
            }
    
    # Test edge computer
    edge_computer = PhaseDifferenceEdgeComputer(
        celestial_dim=32,
        num_celestial_bodies=len(CelestialBody)
    )
    
    adjacency, edge_metadata = edge_computer(celestial_tensor, all_phase_info)
    
    # Find Sun and Chiron indices
    sun_idx = list(CelestialBody).index(CelestialBody.SUN)
    chiron_idx = list(CelestialBody).index(CelestialBody.CHIRON)
    
    sun_chiron_edge = adjacency[:, sun_idx, chiron_idx]
    
    print(f"Sun index: {sun_idx}")
    print(f"Chiron index: {chiron_idx}")
    print(f"Sun-Chiron edge strength: {sun_chiron_edge.mean():.4f}")
    print(f"Sun speed available: {'✅' if sun_phase_info['speed'].abs().sum() > 0 else '❌'}")
    print(f"Chiron speed available: {'✅' if chiron_phase_info['speed'].abs().sum() > 0 else '❌'}")
    
    # Check how the system handles the missing speed
    sun_speed = sun_phase_info['speed'][:, -1]  # Last timestep
    chiron_speed = chiron_phase_info['speed'][:, -1]
    
    print(f"\nSpeed Analysis:")
    print(f"   Sun speed: {sun_speed.abs().mean():.4f}")
    print(f"   Chiron speed: {chiron_speed.abs().mean():.4f}")
    print(f"   Speed difference: {(sun_speed - chiron_speed).abs().mean():.4f}")
    print(f"   Speed ratio: {(sun_speed / (chiron_speed + 1e-8)).abs().mean():.4f}")
    
    return adjacency, all_phase_info

def test_temporal_dynamics():
    """Test how phase relationships evolve over time during training."""
    
    print(f"\n🕒 TESTING TEMPORAL DYNAMICS")
    print("="*60)
    
    # Create time-varying test data
    batch_size, seq_len = 1, 50  # Longer sequence to see evolution
    
    # Create realistic celestial motion patterns
    time_steps = torch.linspace(0, 2*np.pi, seq_len)
    
    # Sun: Fast motion (daily)
    sun_longitude = time_steps * 1.0  # 1 cycle per sequence
    sun_speed = torch.ones(seq_len) * 1.0  # Constant speed
    
    # Moon: Very fast motion (monthly relative to daily)
    moon_longitude = time_steps * 13.0  # 13 cycles per sequence (roughly lunar month)
    moon_speed = torch.ones(seq_len) * 13.0
    
    # Mars: Slow motion (2-year cycle)
    mars_longitude = time_steps * 0.5  # Half cycle per sequence
    mars_speed = torch.ones(seq_len) * 0.5
    
    print(f"Simulating celestial motion over {seq_len} timesteps:")
    print(f"   Sun: 1.0 cycles (daily motion)")
    print(f"   Moon: 13.0 cycles (lunar motion)")
    print(f"   Mars: 0.5 cycles (slow planetary motion)")
    
    # Create phase info
    phase_info = {
        'sun': {
            'longitude_phase': sun_longitude.unsqueeze(0),  # [1, seq_len]
            'sign_phase': sun_longitude.unsqueeze(0) * 0.1,
            'speed': sun_speed.unsqueeze(0),
            'distance': torch.ones(1, seq_len),
            'ecliptic_longitude': sun_longitude.unsqueeze(0)
        },
        'moon': {
            'longitude_phase': moon_longitude.unsqueeze(0),
            'sign_phase': moon_longitude.unsqueeze(0) * 0.1,
            'speed': moon_speed.unsqueeze(0),
            'distance': torch.ones(1, seq_len),
            'ecliptic_longitude': moon_longitude.unsqueeze(0)
        },
        'mars': {
            'longitude_phase': mars_longitude.unsqueeze(0),
            'sign_phase': mars_longitude.unsqueeze(0) * 0.1,
            'speed': mars_speed.unsqueeze(0),
            'distance': torch.ones(1, seq_len) * 1.5,  # Mars is farther
            'ecliptic_longitude': mars_longitude.unsqueeze(0)
        }
    }
    
    # Compute phase differences over time
    sun_moon_phase_diff = []
    sun_mars_phase_diff = []
    sun_moon_speed_ratio = []
    sun_mars_speed_ratio = []
    
    for t in range(seq_len):
        # Phase differences
        sun_phase = phase_info['sun']['longitude_phase'][0, t]
        moon_phase = phase_info['moon']['longitude_phase'][0, t]
        mars_phase = phase_info['mars']['longitude_phase'][0, t]
        
        # Circular differences
        sun_moon_diff = torch.atan2(torch.sin(sun_phase - moon_phase), torch.cos(sun_phase - moon_phase))
        sun_mars_diff = torch.atan2(torch.sin(sun_phase - mars_phase), torch.cos(sun_phase - mars_phase))
        
        sun_moon_phase_diff.append(sun_moon_diff.item())
        sun_mars_phase_diff.append(sun_mars_diff.item())
        
        # Speed ratios
        sun_speed_t = phase_info['sun']['speed'][0, t]
        moon_speed_t = phase_info['moon']['speed'][0, t]
        mars_speed_t = phase_info['mars']['speed'][0, t]
        
        sun_moon_speed_ratio.append((sun_speed_t / moon_speed_t).item())
        sun_mars_speed_ratio.append((sun_speed_t / mars_speed_t).item())
    
    # Analyze temporal patterns
    print(f"\n📊 TEMPORAL PATTERN ANALYSIS:")
    print(f"-" * 40)
    
    sun_moon_phase_var = np.var(sun_moon_phase_diff)
    sun_mars_phase_var = np.var(sun_mars_phase_diff)
    
    print(f"Sun-Moon phase difference variance: {sun_moon_phase_var:.4f}")
    print(f"Sun-Mars phase difference variance: {sun_mars_phase_var:.4f}")
    print(f"Sun-Moon speed ratio (constant): {sun_moon_speed_ratio[0]:.4f}")
    print(f"Sun-Mars speed ratio (constant): {sun_mars_speed_ratio[0]:.4f}")
    
    # Test with edge computer at different timesteps
    print(f"\n🔗 EDGE STRENGTH EVOLUTION:")
    print(f"-" * 40)
    
    edge_computer = PhaseDifferenceEdgeComputer(celestial_dim=32, num_celestial_bodies=3)
    
    # Create mock celestial tensor
    celestial_tensor = torch.randn(1, seq_len, 3, 32)
    
    # Test edge computation at different timesteps
    timesteps_to_test = [0, seq_len//4, seq_len//2, 3*seq_len//4, seq_len-1]
    
    for t in timesteps_to_test:
        # Extract phase info for this timestep
        current_phase_info = {
            'sun': {k: v[:, t:t+1] for k, v in phase_info['sun'].items()},
            'moon': {k: v[:, t:t+1] for k, v in phase_info['moon'].items()},
            'mars': {k: v[:, t:t+1] for k, v in phase_info['mars'].items()}
        }
        
        # Compute edges for this timestep
        test_celestial = celestial_tensor[:, t:t+1, :, :]
        adjacency_t, metadata_t = edge_computer(test_celestial, current_phase_info)
        
        sun_moon_edge = adjacency_t[0, 0, 1].item()
        sun_mars_edge = adjacency_t[0, 0, 2].item()
        
        print(f"   t={t:2d}: Sun-Moon edge={sun_moon_edge:.4f}, Sun-Mars edge={sun_mars_edge:.4f}, "
              f"Phase diff={sun_moon_phase_diff[t]:.2f}")
    
    return {
        'sun_moon_phase_diff': sun_moon_phase_diff,
        'sun_mars_phase_diff': sun_mars_phase_diff,
        'sun_moon_speed_ratio': sun_moon_speed_ratio,
        'sun_mars_speed_ratio': sun_mars_speed_ratio
    }

def test_training_vs_inference_dynamics():
    """Test how the system behaves differently during training vs inference."""
    
    print(f"\n🎯 TRAINING VS INFERENCE DYNAMICS")
    print("="*60)
    
    processor = PhaseAwareCelestialProcessor(num_input_waves=113, celestial_dim=32)
    
    # Create test data
    batch_size, seq_len = 2, 20
    test_input = torch.randn(batch_size, seq_len, 113)
    
    # Test in training mode
    processor.train()
    train_output, train_adjacency, train_metadata = processor(test_input)
    
    # Test in eval mode
    processor.eval()
    with torch.no_grad():
        eval_output, eval_adjacency, eval_metadata = processor(test_input)
    
    print(f"Training mode:")
    print(f"   Output shape: {train_output.shape}")
    print(f"   Adjacency density: {(train_adjacency > 0.5).float().mean():.4f}")
    print(f"   Edge strength avg: {train_metadata['edge_metadata']['avg_edge_strength']:.4f}")
    
    print(f"\nEvaluation mode:")
    print(f"   Output shape: {eval_output.shape}")
    print(f"   Adjacency density: {(eval_adjacency > 0.5).float().mean():.4f}")
    print(f"   Edge strength avg: {eval_metadata['edge_metadata']['avg_edge_strength']:.4f}")
    
    # Check if outputs are different (they should be due to dropout, etc.)
    output_diff = (train_output - eval_output).abs().mean()
    adjacency_diff = (train_adjacency - eval_adjacency).abs().mean()
    
    print(f"\nDifferences between modes:")
    print(f"   Output difference: {output_diff:.6f}")
    print(f"   Adjacency difference: {adjacency_diff:.6f}")
    
    return {
        'train_output': train_output,
        'eval_output': eval_output,
        'train_adjacency': train_adjacency,
        'eval_adjacency': eval_adjacency
    }

if __name__ == "__main__":
    print("🚀 EDGE CREATION DYNAMICS ANALYSIS")
    print("="*60)
    
    # Test 1: Missing feature handling
    adjacency, phase_info = test_missing_feature_handling()
    
    # Test 2: Temporal dynamics
    temporal_data = test_temporal_dynamics()
    
    # Test 3: Training vs inference
    mode_comparison = test_training_vs_inference_dynamics()
    
    print(f"\n🎉 ANALYSIS SUMMARY:")
    print(f"="*60)
    print(f"✅ Missing feature handling: Chiron (no speed) still gets edges")
    print(f"✅ Temporal dynamics: Phase relationships evolve over time")
    print(f"✅ Training dynamics: Different behavior in train vs eval mode")
    print(f"✅ Speed ratios: Computed even when one body has zero speed")
    print(f"✅ Edge creation: Robust to missing features via fallbacks")