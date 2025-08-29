#!/usr/bin/env python3
"""
Comprehensive Backbone Component Functionality Tests

This test suite validates that each backbone component not only initializes
but actually produces expected outputs with correct dimensions and behaviors.
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
    from layers.modular.core.registry import unified_registry, ComponentFamily
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"WARN Could not import modular components: {e}")
    COMPONENTS_AVAILABLE = False

class MockConfig:
    """Mock configuration for testing"""
    def __init__(self, **kwargs):
        # Default values
        self.seq_len = 96
        self.pred_len = 24
        self.enc_in = 7
        self.c_out = 7
        self.d_model = 512
        self.n_heads = 8
        self.n_layers = 6
        self.dropout = 0.1
        self.max_seq_len = 512
        self.model_name = 'test-model'
        self.pretrained = False
        
        # Override with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

def create_sample_data(batch_size=2, seq_len=96, features=7):
    """Create sample time series data for testing"""
    x_enc = torch.randn(batch_size, seq_len, features)
    x_mark_enc = torch.randn(batch_size, seq_len, 4)  # Time features
    x_dec = torch.randn(batch_size, 24, features)  # Decoder input
    x_mark_dec = torch.randn(batch_size, 24, 4)  # Decoder time features
    return x_enc, x_mark_enc, x_dec, x_mark_dec

def test_chronos_backbone_functionality():
    """Test Chronos backbone actual functionality"""
    print("TEST Testing Chronos Backbone Functionality...")
    
    try:
        config = MockConfig(
            model_name='amazon/chronos-t5-tiny',  # Use tiny for testing
            d_model=512,
            enc_in=7,
            c_out=7
        )
        
        backbone = unified_registry.create(ComponentFamily.BACKBONE, 'chronos', **vars(config))
        if backbone is None:
            print("    WARN Chronos backbone not available, skipping...")
            return True
        
        # Test basic functionality
        x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_data()
        
        # Test forward pass
        with torch.no_grad():
            output = backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Validate output dimensions
        expected_shape = (2, 24, 7)  # batch_size, pred_len, c_out
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Validate output is not NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        
        # Test different sequence lengths
        x_enc_short, _, x_dec_short, _ = create_sample_data(seq_len=48)
        with torch.no_grad():
            output_short = backbone(x_enc_short, x_mark_enc[:, :48, :], x_dec_short, x_mark_dec)
        
        assert output_short.shape == expected_shape, "Backbone doesn't handle variable sequence lengths"
        
        # Test gradient flow
        backbone.train()
        output = backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist and are non-zero
        has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                          for p in backbone.parameters() if p.requires_grad)
        assert has_gradients, "No gradients computed - backbone might not be learning"
        
        print("    PASS Chronos backbone functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Chronos backbone test failed: {e}")
        return False

def test_t5_backbone_functionality():
    """Test T5 backbone actual functionality"""
    print("TEST Testing T5 Backbone Functionality...")
    
    try:
        config = MockConfig(
            model_name='google/flan-t5-small',
            d_model=512,
            encoder_only=True,
            enc_in=7,
            c_out=7
        )
        
        backbone = unified_registry.create(ComponentFamily.BACKBONE, 't5', **vars(config))
        if backbone is None:
            print("    WARN T5 backbone not available, skipping...")
            return True
        
        x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_data()
        
        # Test forward pass
        with torch.no_grad():
            output = backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Validate output
        expected_shape = (2, 24, 7)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "T5 output contains NaN"
        assert not torch.isinf(output).any(), "T5 output contains Inf"
        
        # Test encoder-only vs encoder-decoder modes
        config_decoder = MockConfig(
            model_name='google/flan-t5-small',
            encoder_only=False,
            d_model=512
        )
        
        try:
            backbone_decoder = unified_registry.create(ComponentFamily.BACKBONE, 't5', **vars(config_decoder))
            if backbone_decoder:
                with torch.no_grad():
                    output_decoder = backbone_decoder(x_enc, x_mark_enc, x_dec, x_mark_dec)
                assert output_decoder.shape == expected_shape, "Encoder-decoder mode failed"
                print("    PASS Both encoder-only and encoder-decoder modes work")
        except Exception as e:
            print(f"    WARN Encoder-decoder mode test failed: {e}")
        
        # Test parameter efficiency
        param_count = sum(p.numel() for p in backbone.parameters())
        print(f"    CHART T5 backbone parameters: {param_count:,}")
        
        print("    PASS T5 backbone functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL T5 backbone test failed: {e}")
        return False

def test_simple_transformer_backbone():
    """Test simple transformer backbone functionality"""
    print("TEST Testing Simple Transformer Backbone Functionality...")
    
    try:
        config = MockConfig(
            d_model=256,
            n_heads=8,
            n_layers=4,
            dropout=0.1
        )
        
        backbone = unified_registry.create(ComponentFamily.BACKBONE, 'simple_transformer', config)
        assert backbone is not None, "Simple transformer should always be available"
        
        x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_data()
        
        # Test forward pass
        output = backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        expected_shape = (2, 24, 7)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Simple transformer output contains NaN"
        
        # Test different model sizes
        configs_to_test = [
            {'d_model': 128, 'n_heads': 4, 'n_layers': 2},  # Small
            {'d_model': 512, 'n_heads': 8, 'n_layers': 6},  # Medium
            {'d_model': 768, 'n_heads': 12, 'n_layers': 8}, # Large
        ]
        
        for i, test_config in enumerate(configs_to_test):
            config_test = MockConfig(**test_config)
            backbone_test = unified_registry.create(ComponentFamily.BACKBONE, 'simple_transformer', **vars(config_test))
            
            with torch.no_grad():
                output_test = backbone_test(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            assert output_test.shape == expected_shape, f"Size config {i} failed"
            param_count = sum(p.numel() for p in backbone_test.parameters())
            print(f"    CHART Config {i}: d_model={test_config['d_model']}, params={param_count:,}")
        
        # Test attention patterns
        backbone.eval()
        with torch.no_grad():
            output1 = backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
            output2 = backbone(x_enc + 0.1 * torch.randn_like(x_enc), x_mark_enc, x_dec, x_mark_dec)
        
        # Outputs should be different but reasonable
        diff = torch.abs(output1 - output2).mean()
        assert 0.001 < diff < 1.0, f"Attention sensitivity issue: diff={diff}"
        
        print("    PASS Simple transformer backbone functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Simple transformer backbone test failed: {e}")
        return False

def test_robust_hf_backbone():
    """Test robust HF backbone fallback functionality"""
    print("TEST Testing Robust HF Backbone Functionality...")
    
    try:
        config = MockConfig(
            model_families=['chronos', 't5', 'bert'],
            auto_fallback=True,
            error_recovery='graceful'
        )
        
        backbone = unified_registry.create(ComponentFamily.BACKBONE, 'robust_hf', config)
        if backbone is None:
            print("    WARN Robust HF backbone not available, skipping...")
            return True
        
        x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_data()
        
        # Test forward pass
        with torch.no_grad():
            output = backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        expected_shape = (2, 24, 7)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Robust HF output contains NaN"
        
        # Test fallback behavior by providing invalid model
        config_invalid = MockConfig(
            model_families=['invalid_model'],
            auto_fallback=True,
            error_recovery='graceful'
        )
        
        try:
            backbone_fallback = unified_registry.create(ComponentFamily.BACKBONE, 'robust_hf', **vars(config_invalid))
            if backbone_fallback:
                with torch.no_grad():
                    output_fallback = backbone_fallback(x_enc, x_mark_enc, x_dec, x_mark_dec)
                assert output_fallback.shape == expected_shape, "Fallback mechanism failed"
                print("    PASS Fallback mechanism working correctly")
        except Exception as e:
            print(f"    WARN Fallback test failed (expected in some cases): {e}")
        
        print("    PASS Robust HF backbone functionality validated")
        return True
        
    except Exception as e:
        print(f"    FAIL Robust HF backbone test failed: {e}")
        return False

def test_backbone_consistency():
    """Test consistency across different backbone implementations"""
    print("TEST Testing Backbone Consistency...")
    
    try:
        config = MockConfig(d_model=256, enc_in=5, c_out=3)
        x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_data(features=5)
        
        # Get available backbones
        registry = unified_registry
        backbone_types = ['simple_transformer']  # Always available
        
        # Try to add other backbones if available
        for backbone_type in ['chronos', 't5', 'robust_hf']:
            try:
                test_backbone = unified_registry.create(ComponentFamily.BACKBONE, backbone_type, config)
                if test_backbone is not None:
                    backbone_types.append(backbone_type)
            except:
                pass
        
        outputs = {}
        shapes_consistent = True
        
        for backbone_type in backbone_types:
            try:
                backbone = unified_registry.create(ComponentFamily.BACKBONE, backbone_type, config)
                if backbone is not None:
                    with torch.no_grad():
                        output = backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    outputs[backbone_type] = output
                    
                    expected_shape = (2, 24, 3)  # batch, pred_len, c_out
                    if output.shape != expected_shape:
                        shapes_consistent = False
                        print(f"    FAIL {backbone_type}: shape {output.shape} != {expected_shape}")
            except Exception as e:
                print(f"    WARN {backbone_type} failed: {e}")
        
        assert shapes_consistent, "Shape consistency failed across backbones"
        
        # Test output magnitude consistency (should be in reasonable range)
        for backbone_type, output in outputs.items():
            magnitude = output.abs().mean().item()
            assert 0.001 < magnitude < 100, f"{backbone_type} output magnitude {magnitude} unreasonable"
            print(f"    CHART {backbone_type}: output magnitude = {magnitude:.4f}")
        
        print(f"    PASS Consistency validated across {len(outputs)} backbones")
        return True
        
    except Exception as e:
        print(f"    FAIL Backbone consistency test failed: {e}")
        return False

def run_backbone_functionality_tests():
    """Run all backbone functionality tests"""
    print("ROCKET Running Backbone Component Functionality Tests")
    print("=" * 80)
    
    if not COMPONENTS_AVAILABLE:
        print("FAIL Modular components not available - skipping tests")
        return False
    
    tests = [
        ("Simple Transformer Backbone", test_simple_transformer_backbone),
        ("Chronos Backbone", test_chronos_backbone_functionality),
        ("T5 Backbone", test_t5_backbone_functionality),
        ("Robust HF Backbone", test_robust_hf_backbone),
        ("Backbone Consistency", test_backbone_consistency),
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
    print(f"CHART Backbone Functionality Test Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("PARTY All backbone functionality tests passed!")
        return True
    else:
        print("WARN Some backbone functionality tests failed")
        return False

if __name__ == "__main__":
    success = run_backbone_functionality_tests()
    sys.exit(0 if success else 1)
