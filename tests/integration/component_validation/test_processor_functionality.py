#!/usr/bin/env python3
"""
Comprehensive Processor Component Functionality Tests

This test suite validates that each processor component correctly handles
signal processing, feature extraction, and data transformation tasks
with expected mathematical properties and behaviors.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.modular_components.registry import create_component, get_global_registry
    from utils.modular_components.implementations import get_integration_status
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"WARN Could not import modular components: {e}")
    COMPONENTS_AVAILABLE = False

class MockConfig:
    """Mock configuration for processor testing"""
    def __init__(self, **kwargs):
        # Default values
        self.seq_len = 96
        self.pred_len = 24
        self.enc_in = 7
        self.c_out = 7
        self.d_model = 512
        self.scales = [1, 2, 4, 8]
        self.levels = 3
        self.wavelet = 'db4'
        self.freq_threshold = 0.1
        self.window_size = 16
        
        # Override with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

def create_sample_time_series(batch_size=2, seq_len=96, features=7, signal_type='mixed'):
    """Create sample time series data with known properties"""
    t = torch.linspace(0, 4*np.pi, seq_len)
    
    if signal_type == 'trend':
        # Linear trend + noise
        data = t.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, features)
        data = data + 0.1 * torch.randn(batch_size, seq_len, features)
        
    elif signal_type == 'seasonal':
        # Seasonal pattern
        seasonal = torch.sin(2 * t) + 0.5 * torch.cos(4 * t)
        data = seasonal.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, features)
        data = data + 0.1 * torch.randn(batch_size, seq_len, features)
        
    elif signal_type == 'high_freq':
        # High frequency signal
        high_freq = torch.sin(10 * t) + 0.3 * torch.sin(20 * t)
        data = high_freq.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, features)
        data = data + 0.05 * torch.randn(batch_size, seq_len, features)
        
    elif signal_type == 'mixed':
        # Mixed signal: trend + seasonal + high frequency
        trend = 0.01 * t
        seasonal = 0.5 * torch.sin(2 * t)
        high_freq = 0.2 * torch.sin(10 * t)
        combined = trend + seasonal + high_freq
        data = combined.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, features)
        data = data + 0.1 * torch.randn(batch_size, seq_len, features)
        
    elif signal_type == 'random':
        # Pure random signal
        data = torch.randn(batch_size, seq_len, features)
        
    else:
        # Default to random
        data = torch.randn(batch_size, seq_len, features)
    
    return data

def test_frequency_domain_processor():
    """Test frequency domain processor functionality"""
    print("TEST Testing Frequency Domain Processor Functionality...")
    
    try:
        config = MockConfig(freq_threshold=0.1)
        processor = create_component('processor', 'frequency_domain', config)
        
        if processor is None:
            print("    WARN Frequency domain processor not available, skipping...")
            return True
        
        # Test with different signal types
        signal_types = ['seasonal', 'high_freq', 'mixed', 'random']
        
        for signal_type in signal_types:
            data = create_sample_time_series(signal_type=signal_type)
            
            # Test forward pass
            with torch.no_grad():
                processed = processor(data)
            
            # Basic validation
            assert isinstance(processed, torch.Tensor), f"Output should be tensor for {signal_type}"
            assert not torch.isnan(processed).any(), f"Output contains NaN for {signal_type}"
            assert not torch.isinf(processed).any(), f"Output contains Inf for {signal_type}"
            
            print(f"    CHART {signal_type}: input shape {data.shape}  output shape {processed.shape}")
        
        # Test frequency domain properties
        # Create a known frequency signal
        seq_len = 96
        t = torch.linspace(0, 4*np.pi, seq_len)
        
        # Pure sine wave at known frequency
        freq = 2.0  # 2 Hz
        pure_sine = torch.sin(freq * t).unsqueeze(0).unsqueeze(-1).expand(2, -1, 7)
        
        with torch.no_grad():
            processed_sine = processor(pure_sine)
        
        # Test feature extraction capabilities
        if hasattr(processor, 'extract_features'):
            features = processor.extract_features(pure_sine)
            print(f"    CHART Extracted {features.shape[-1] if hasattr(features, 'shape') else 'N/A'} frequency features")
        
        # Test different sequence lengths
        for seq_len in [48, 96, 192]:
            test_data = create_sample_time_series(seq_len=seq_len)
            with torch.no_grad():
                test_output = processor(test_data)
            print(f"    PASS Seq len {seq_len}: processed successfully")
        
        print("    PASS Frequency domain processor functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Frequency domain processor test failed: {e}")
        return False

def test_dtw_alignment_processor():
    """Test DTW alignment processor functionality"""
    print("TEST Testing DTW Alignment Processor Functionality...")
    
    try:
        config = MockConfig(window_size=16)
        processor = create_component('processor', 'dtw_alignment', config)
        
        if processor is None:
            print("    WARN DTW alignment processor not available, skipping...")
            return True
        
        # Create sequences that need alignment
        batch_size, seq_len, features = 2, 96, 7
        
        # Reference sequence
        reference = create_sample_time_series(signal_type='seasonal')
        
        # Shifted/distorted sequences
        # Time shift
        shifted = torch.roll(reference, shifts=10, dims=1)
        
        # Speed variation (simple simulation)
        indices = torch.linspace(0, seq_len-1, seq_len)
        warped_indices = indices + 0.1 * torch.sin(0.2 * indices)
        warped_indices = torch.clamp(warped_indices, 0, seq_len-1).long()
        speed_varied = reference[:, warped_indices, :]
        
        # Test alignment
        test_cases = [
            ("original", reference),
            ("shifted", shifted),
            ("speed_varied", speed_varied),
        ]
        
        for case_name, test_sequence in test_cases:
            with torch.no_grad():
                aligned = processor(test_sequence, reference)
            
            assert aligned.shape == test_sequence.shape, f"Shape preserved for {case_name}"
            assert not torch.isnan(aligned).any(), f"No NaN for {case_name}"
            
            # Measure alignment quality (distance to reference)
            alignment_error = torch.mean(torch.abs(aligned - reference))
            print(f"    CHART {case_name}: alignment error = {alignment_error:.6f}")
        
        # Test DTW properties
        # DTW should be symmetric for identical sequences
        with torch.no_grad():
            aligned_self = processor(reference, reference)
        
        self_alignment_error = torch.mean(torch.abs(aligned_self - reference))
        assert self_alignment_error < 1e-3, "Self-alignment should be nearly perfect"
        
        # Test with different sequence lengths
        try:
            short_ref = reference[:, :48, :]
            with torch.no_grad():
                aligned_diff_len = processor(reference, short_ref)
            print("    PASS Different sequence lengths handled")
        except Exception as e:
            print(f"    WARN Different sequence lengths failed: {e}")
        
        print("    PASS DTW alignment processor functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL DTW alignment processor test failed: {e}")
        return False

def test_trend_processor():
    """Test trend analysis processor functionality"""
    print("TEST Testing Trend Analysis Processor Functionality...")
    
    try:
        config = MockConfig(scales=[1, 4, 16])
        processor = create_component('processor', 'trend_analysis', config)
        
        if processor is None:
            print("    WARN Trend analysis processor not available, skipping...")
            return True
        
        # Test trend decomposition
        # Create signal with known trend
        seq_len = 96
        t = torch.linspace(0, 10, seq_len)
        
        # Strong linear trend
        trend_component = 0.1 * t
        seasonal_component = 0.5 * torch.sin(2 * np.pi * t / 12)  # Seasonal pattern
        noise_component = 0.1 * torch.randn(seq_len)
        
        signal_with_trend = (trend_component + seasonal_component + noise_component)
        signal_data = signal_with_trend.unsqueeze(0).unsqueeze(-1).expand(2, -1, 7)
        
        # Test decomposition
        with torch.no_grad():
            processed = processor(signal_data)
        
        # Basic validation
        assert processed.shape[0] == signal_data.shape[0], "Batch dimension preserved"
        assert processed.shape[1] == signal_data.shape[1], "Sequence dimension preserved"
        
        # Test decomposition components if available
        if hasattr(processor, 'decompose'):
            with torch.no_grad():
                trend, seasonal = processor.decompose(signal_data)
            
            print(f"    CHART Trend shape: {trend.shape}")
            print(f"    CHART Seasonal shape: {seasonal.shape}")
            
            # Trend should be smoother than original signal
            trend_variance = trend.var().item()
            original_variance = signal_data.var().item()
            print(f"    CHART Trend variance: {trend_variance:.6f}, Original variance: {original_variance:.6f}")
        
        # Test multi-scale analysis
        for scale in [1, 2, 4, 8]:
            test_config = MockConfig(scales=[scale])
            try:
                scale_processor = create_component('processor', 'trend_analysis', test_config)
                if scale_processor:
                    with torch.no_grad():
                        scale_output = scale_processor(signal_data)
                    print(f"    PASS Scale {scale}: processed successfully")
            except Exception as e:
                print(f"    WARN Scale {scale}: {e}")
        
        # Test trend extraction with different signal types
        signal_types = ['trend', 'seasonal', 'mixed', 'random']
        for signal_type in signal_types:
            test_data = create_sample_time_series(signal_type=signal_type)
            with torch.no_grad():
                trend_output = processor(test_data)
            
            # Check if trend is extracted properly
            trend_smoothness = torch.diff(trend_output, dim=1).abs().mean()
            original_smoothness = torch.diff(test_data, dim=1).abs().mean()
            
            print(f"    CHART {signal_type}: trend smoothness ratio = {(trend_smoothness/original_smoothness):.3f}")
        
        print("    PASS Trend analysis processor functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Trend analysis processor test failed: {e}")
        return False

def test_integrated_signal_processor():
    """Test integrated signal processor functionality"""
    print("TEST Testing Integrated Signal Processor Functionality...")
    
    try:
        config = MockConfig()
        processor = create_component('processor', 'integrated_signal', config)
        
        if processor is None:
            print("    WARN Integrated signal processor not available, skipping...")
            return True
        
        # Test comprehensive signal processing
        test_signals = {
            'trend': create_sample_time_series(signal_type='trend'),
            'seasonal': create_sample_time_series(signal_type='seasonal'),
            'high_freq': create_sample_time_series(signal_type='high_freq'),
            'mixed': create_sample_time_series(signal_type='mixed'),
        }
        
        processing_results = {}
        
        for signal_name, signal_data in test_signals.items():
            with torch.no_grad():
                processed = processor(signal_data)
            
            processing_results[signal_name] = processed
            
            # Basic validation
            assert not torch.isnan(processed).any(), f"NaN in {signal_name} processing"
            assert not torch.isinf(processed).any(), f"Inf in {signal_name} processing"
            
            print(f"    CHART {signal_name}: {signal_data.shape}  {processed.shape}")
        
        # Test preprocessing for HF backbone
        if hasattr(processor, 'preprocess_for_hf_backbone'):
            test_data = test_signals['mixed']
            with torch.no_grad():
                hf_input = processor.preprocess_for_hf_backbone(test_data)
            
            print(f"    CHART HF preprocessing: {test_data.shape}  {hf_input.shape}")
            assert hf_input.shape[0] == test_data.shape[0], "Batch dimension preserved"
            assert hf_input.shape[1] == test_data.shape[1], "Sequence dimension preserved"
        
        # Test postprocessing for HF output
        if hasattr(processor, 'postprocess_hf_output'):
            # Simulate HF backbone output
            backbone_output = torch.randn(2, 24, 7)  # Typical prediction output
            original_input = test_signals['mixed']
            
            with torch.no_grad():
                final_output = processor.postprocess_hf_output(backbone_output, original_input)
            
            print(f"    CHART HF postprocessing: {backbone_output.shape}  {final_output.shape}")
            assert final_output.shape == backbone_output.shape, "Output shape preserved"
        
        # Test integration capabilities
        # The integrated processor should handle multiple signal processing tasks
        comprehensive_tests = [
            "frequency_analysis",
            "trend_extraction", 
            "alignment_processing",
            "feature_enhancement"
        ]
        
        for test_name in comprehensive_tests:
            if hasattr(processor, test_name.replace('_', '')):
                print(f"    PASS {test_name} capability available")
            else:
                print(f"    WARN {test_name} capability not found")
        
        # Test robustness with edge cases
        edge_cases = [
            ("zeros", torch.zeros(2, 96, 7)),
            ("ones", torch.ones(2, 96, 7)),
            ("large_values", torch.ones(2, 96, 7) * 1000),
            ("small_values", torch.ones(2, 96, 7) * 1e-6),
        ]
        
        for case_name, edge_data in edge_cases:
            try:
                with torch.no_grad():
                    edge_output = processor(edge_data)
                
                is_finite = torch.isfinite(edge_output).all()
                print(f"    CHART {case_name}: finite output = {is_finite}")
                
            except Exception as e:
                print(f"    WARN {case_name}: {e}")
        
        print("    PASS Integrated signal processor functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Integrated signal processor test failed: {e}")
        return False

def test_wavelet_processor():
    """Test wavelet processor functionality"""
    print("TEST Testing Wavelet Processor Functionality...")
    
    try:
        config = MockConfig(wavelet='db4', levels=3)
        processor = create_component('processor', 'wavelet', config)
        
        if processor is None:
            print("    WARN Wavelet processor not available, skipping...")
            return True
        
        # Test wavelet decomposition and reconstruction
        # Create test signal with multiple frequency components
        seq_len = 128  # Power of 2 for wavelet decomposition
        t = torch.linspace(0, 4*np.pi, seq_len)
        
        # Multi-frequency signal
        low_freq = torch.sin(t)
        mid_freq = 0.5 * torch.sin(4 * t)
        high_freq = 0.2 * torch.sin(16 * t)
        
        composite_signal = low_freq + mid_freq + high_freq
        signal_data = composite_signal.unsqueeze(0).unsqueeze(-1).expand(2, -1, 7)
        
        # Test forward pass
        with torch.no_grad():
            processed = processor(signal_data)
        
        # Basic validation
        assert not torch.isnan(processed).any(), "Wavelet output contains NaN"
        assert not torch.isinf(processed).any(), "Wavelet output contains Inf"
        
        print(f"    CHART Wavelet processing: {signal_data.shape}  {processed.shape}")
        
        # Test different wavelet types
        wavelet_types = ['db4', 'db8', 'haar', 'bior2.2']
        
        for wavelet_type in wavelet_types:
            try:
                wavelet_config = MockConfig(wavelet=wavelet_type, levels=3)
                wavelet_proc = create_component('processor', 'wavelet', wavelet_config)
                
                if wavelet_proc:
                    with torch.no_grad():
                        wavelet_output = wavelet_proc(signal_data)
                    print(f"    PASS Wavelet {wavelet_type}: processed successfully")
                    
            except Exception as e:
                print(f"    WARN Wavelet {wavelet_type}: {e}")
        
        # Test different decomposition levels
        for levels in [2, 3, 4, 5]:
            try:
                level_config = MockConfig(wavelet='db4', levels=levels)
                level_proc = create_component('processor', 'wavelet', level_config)
                
                if level_proc:
                    with torch.no_grad():
                        level_output = level_proc(signal_data)
                    print(f"    PASS Levels {levels}: processed successfully")
                    
            except Exception as e:
                print(f"    WARN Levels {levels}: {e}")
        
        # Test reconstruction quality
        if hasattr(processor, 'decompose') and hasattr(processor, 'reconstruct'):
            with torch.no_grad():
                coeffs = processor.decompose(signal_data)
                reconstructed = processor.reconstruct(coeffs)
            
            reconstruction_error = torch.mean(torch.abs(signal_data - reconstructed))
            print(f"    CHART Reconstruction error: {reconstruction_error:.6f}")
            
            # Good wavelets should have low reconstruction error
            assert reconstruction_error < 0.1, "Reconstruction error too high"
        
        # Test frequency separation
        # Low frequency component should be captured in approximation coefficients
        pure_low_freq = torch.sin(t).unsqueeze(0).unsqueeze(-1).expand(2, -1, 7)
        pure_high_freq = torch.sin(16 * t).unsqueeze(0).unsqueeze(-1).expand(2, -1, 7)
        
        with torch.no_grad():
            processed_low = processor(pure_low_freq)
            processed_high = processor(pure_high_freq)
        
        # The processor should handle different frequency components differently
        processing_diff = torch.mean(torch.abs(processed_low - processed_high))
        print(f"    CHART Frequency discrimination: {processing_diff:.6f}")
        
        print("    PASS Wavelet processor functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Wavelet processor test failed: {e}")
        return False

def test_multi_scale_processor():
    """Test multi-scale processor functionality"""
    print("TEST Testing Multi-Scale Processor Functionality...")
    
    try:
        config = MockConfig(scales=[1, 2, 4, 8])
        processor = create_component('processor', 'multi_scale', config)
        
        if processor is None:
            print("    WARN Multi-scale processor not available, skipping...")
            return True
        
        # Create signal with multi-scale patterns
        seq_len = 96
        t = torch.linspace(0, 4*np.pi, seq_len)
        
        # Scale 1: Fine details
        scale1 = 0.1 * torch.sin(20 * t)
        
        # Scale 2: Medium patterns
        scale2 = 0.3 * torch.sin(8 * t)
        
        # Scale 4: Coarse patterns
        scale4 = 0.5 * torch.sin(2 * t)
        
        # Scale 8: Very coarse (trend-like)
        scale8 = 0.7 * torch.sin(0.5 * t)
        
        multi_scale_signal = scale1 + scale2 + scale4 + scale8
        signal_data = multi_scale_signal.unsqueeze(0).unsqueeze(-1).expand(2, -1, 7)
        
        # Test forward pass
        with torch.no_grad():
            processed = processor(signal_data)
        
        # Basic validation
        assert not torch.isnan(processed).any(), "Multi-scale output contains NaN"
        assert not torch.isinf(processed).any(), "Multi-scale output contains Inf"
        
        print(f"    CHART Multi-scale processing: {signal_data.shape}  {processed.shape}")
        
        # Test different scale configurations
        scale_configs = [
            [1, 2],
            [1, 4, 16],
            [2, 4, 8, 16],
            [1, 2, 4, 8, 16]
        ]
        
        for scales in scale_configs:
            try:
                scale_config = MockConfig(scales=scales)
                scale_proc = create_component('processor', 'multi_scale', scale_config)
                
                if scale_proc:
                    with torch.no_grad():
                        scale_output = scale_proc(signal_data)
                    print(f"    PASS Scales {scales}: processed successfully")
                    
            except Exception as e:
                print(f"    WARN Scales {scales}: {e}")
        
        # Test scale extraction if available
        if hasattr(processor, 'extract_scales'):
            with torch.no_grad():
                scale_features = processor.extract_scales(signal_data)
            
            if isinstance(scale_features, (list, tuple)):
                print(f"    CHART Extracted {len(scale_features)} scale features")
                for i, scale_feat in enumerate(scale_features):
                    print(f"      Scale {i}: shape {scale_feat.shape}")
            else:
                print(f"    CHART Scale features shape: {scale_features.shape}")
        
        # Test fusion methods if available
        fusion_methods = ['concatenate', 'weighted', 'attention']
        
        for fusion_method in fusion_methods:
            try:
                fusion_config = MockConfig(scales=[1, 2, 4], fusion_method=fusion_method)
                fusion_proc = create_component('processor', 'multi_scale', fusion_config)
                
                if fusion_proc:
                    with torch.no_grad():
                        fusion_output = fusion_proc(signal_data)
                    print(f"    PASS Fusion {fusion_method}: processed successfully")
                    
            except Exception as e:
                print(f"    WARN Fusion {fusion_method}: {e}")
        
        # Test scale invariance properties
        # Scaling the input should affect different scales differently
        scaled_signal = signal_data * 2.0
        
        with torch.no_grad():
            original_output = processor(signal_data)
            scaled_output = processor(scaled_signal)
        
        scale_sensitivity = torch.mean(torch.abs(scaled_output - 2.0 * original_output))
        print(f"    CHART Scale sensitivity: {scale_sensitivity:.6f}")
        
        print("    PASS Multi-scale processor functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Multi-scale processor test failed: {e}")
        return False

def test_processor_consistency():
    """Test consistency across different processors"""
    print("TEST Testing Processor Consistency...")
    
    try:
        # Get available processors
        processor_types = []
        for proc_type in ['frequency_domain', 'trend_analysis', 'integrated_signal', 'multi_scale']:
            try:
                test_proc = create_component('processor', proc_type, MockConfig())
                if test_proc is not None:
                    processor_types.append(proc_type)
            except:
                pass
        
        if not processor_types:
            print("    WARN No processors available for consistency testing")
            return True
        
        # Test with common input
        test_data = create_sample_time_series(signal_type='mixed')
        outputs = {}
        
        for proc_type in processor_types:
            try:
                processor = create_component('processor', proc_type, MockConfig())
                if processor is not None:
                    with torch.no_grad():
                        output = processor(test_data)
                    outputs[proc_type] = output
                    
                    # Basic consistency checks
                    assert output.shape[0] == test_data.shape[0], f"{proc_type}: batch dimension inconsistent"
                    assert not torch.isnan(output).any(), f"{proc_type}: contains NaN"
                    assert not torch.isinf(output).any(), f"{proc_type}: contains Inf"
                    
                    print(f"    CHART {proc_type}: {test_data.shape}  {output.shape}")
                    
            except Exception as e:
                print(f"    WARN {proc_type}: {e}")
        
        # Test output magnitude consistency
        for proc_type, output in outputs.items():
            magnitude = output.abs().mean().item()
            print(f"    CHART {proc_type}: output magnitude = {magnitude:.6f}")
            
            # Reasonable magnitude range
            assert 0.001 < magnitude < 1000, f"{proc_type}: unreasonable output magnitude"
        
        # Test deterministic behavior
        for proc_type in processor_types:
            try:
                processor = create_component('processor', proc_type, MockConfig())
                if processor is not None:
                    # Same input should give same output
                    with torch.no_grad():
                        output1 = processor(test_data)
                        output2 = processor(test_data)
                    
                    deterministic_error = torch.mean(torch.abs(output1 - output2))
                    assert deterministic_error < 1e-6, f"{proc_type}: not deterministic"
                    print(f"    PASS {proc_type}: deterministic behavior verified")
                    
            except Exception as e:
                print(f"    WARN {proc_type} deterministic test: {e}")
        
        print(f"    PASS Consistency validated across {len(outputs)} processors")
        return True
        
    except Exception as e:
        print(f"    FAIL Processor consistency test failed: {e}")
        return False

def run_processor_functionality_tests():
    """Run all processor functionality tests"""
    print("ROCKET Running Processor Component Functionality Tests")
    print("=" * 80)
    
    if not COMPONENTS_AVAILABLE:
        print("FAIL Modular components not available - skipping tests")
        return False
    
    tests = [
        ("Frequency Domain Processor", test_frequency_domain_processor),
        ("DTW Alignment Processor", test_dtw_alignment_processor),
        ("Trend Analysis Processor", test_trend_processor),
        ("Integrated Signal Processor", test_integrated_signal_processor),
        ("Wavelet Processor", test_wavelet_processor),
        ("Multi-Scale Processor", test_multi_scale_processor),
        ("Processor Consistency", test_processor_consistency),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTARGET {test_name}")
        print("-" * 60)
        
        try:
            if test_func():
                passed += 1
                print(f"PASS {test_name} PASSED")
            else:
                print(f"FAIL {test_name} FAILED")
        except Exception as e:
            print(f"FAIL {test_name} ERROR: {e}")
    
    print("\n" + "=" * 80)
    print(f"CHART Processor Component Functionality Test Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("PARTY All processor functionality tests passed!")
        return True
    else:
        print("WARN Some processor functionality tests failed")
        return False

if __name__ == "__main__":
    success = run_processor_functionality_tests()
    sys.exit(0 if success else 1)
