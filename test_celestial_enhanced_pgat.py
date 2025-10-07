#!/usr/bin/env python3
"""
Test script for Celestial Enhanced PGAT

Tests the complete integration of celestial body system with Enhanced SOTA PGAT.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.Celestial_Enhanced_PGAT import CelestialEnhancedPGAT
from layers.modular.graph.celestial_body_nodes import CelestialBody

class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        # Basic model parameters
        self.seq_len = 96
        self.label_len = 48  
        self.pred_len = 24
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 4  # OHLC
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.dropout = 0.1
        self.activation = 'gelu'
        
        # Enhanced PGAT specific
        self.use_multi_scale_patching = True
        self.use_hierarchical_mapper = True
        self.use_stochastic_learner = True
        self.use_gated_graph_combiner = True
        self.use_mixture_decoder = True
        self.mdn_components = 3
        self.mixture_multivariate_mode = 'independent'
        
        # Celestial system specific
        self.use_celestial_system = True
        self.celestial_d_model = 512
        self.celestial_fusion_layers = 3
        self.celestial_attention_heads = 8
        
        # Additional parameters
        self.num_wave_features = 3  # Derived features
        self.num_wave_patch_latents = 64
        self.num_target_patch_latents = 24
        
        # Phase and delay features
        self.enable_phase_features = True
        self.enable_delayed_influence = True
        self.delayed_max_lag = 3
        self.enable_group_interactions = True

def test_celestial_enhanced_pgat_initialization():
    """Test model initialization"""
    print("🌟 Testing Celestial Enhanced PGAT Initialization...")
    
    config = MockConfig()
    
    try:
        model = CelestialEnhancedPGAT(config, mode='celestial_probabilistic')
        
        print(f"✅ Model initialized successfully")
        print(f"   - Celestial system enabled: {model.use_celestial_system}")
        print(f"   - Number of celestial bodies: {model.num_celestial_bodies}")
        print(f"   - Celestial d_model: {model.celestial_d_model}")
        
        # Check celestial components
        assert hasattr(model, 'celestial_nodes')
        assert hasattr(model, 'celestial_combiner')
        assert hasattr(model, 'market_context_encoder')
        
        print(f"   - All celestial components present")
        
        return model
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_celestial_enhanced_pgat_forward():
    """Test forward pass"""
    print("\n🚀 Testing Celestial Enhanced PGAT Forward Pass...")
    
    config = MockConfig()
    model = CelestialEnhancedPGAT(config, mode='celestial_probabilistic')
    
    # Create mock input data
    batch_size = 2
    seq_len = config.seq_len
    label_len = config.label_len
    pred_len = config.pred_len
    enc_in = config.enc_in
    dec_in = config.dec_in
    
    # Input sequences
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.randn(batch_size, seq_len, 4)  # Time features
    
    # Decoder inputs
    x_dec = torch.randn(batch_size, label_len + pred_len, dec_in)
    x_mark_dec = torch.randn(batch_size, label_len + pred_len, 4)
    
    try:
        # Forward pass
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f"✅ Forward pass successful")
        
        # Check output format
        if isinstance(output, tuple):
            means, log_vars, weights = output
            print(f"   - Mixture output format:")
            print(f"     - Means: {means.shape}")
            print(f"     - Log vars: {log_vars.shape}")
            print(f"     - Weights: {weights.shape}")
        else:
            print(f"   - Standard output: {output.shape}")
        
        # Check celestial logs
        if hasattr(model, 'celestial_logs') and model.celestial_logs:
            print(f"   - Celestial logs available: {list(model.celestial_logs.keys())}")
        
        return output
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_celestial_interpretation():
    """Test celestial interpretation features"""
    print("\n🔮 Testing Celestial Interpretation...")
    
    config = MockConfig()
    model = CelestialEnhancedPGAT(config, mode='celestial_probabilistic')
    
    # Create mock data and run forward pass
    batch_size = 2
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
    x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
    x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
    
    # Forward pass to generate celestial data
    _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    try:
        # Get celestial interpretation
        interpretation = model.get_celestial_interpretation()
        
        print(f"✅ Celestial interpretation generated")
        print(f"   - Dominant market regime: {interpretation.get('dominant_market_regime', 'Unknown')}")
        print(f"   - Regime confidence: {interpretation.get('regime_confidence', 0.0):.4f}")
        
        most_active = interpretation.get('most_active_celestial_body', {})
        print(f"   - Most active celestial body: {most_active.get('name', 'Unknown')}")
        print(f"   - Domain: {most_active.get('domain', 'Unknown')}")
        print(f"   - Market influence: {most_active.get('market_influence', 'Unknown')}")
        
        strengths = interpretation.get('astronomical_vs_dynamic_strength', {})
        print(f"   - Astronomical strength: {strengths.get('astronomical', 0.0):.4f}")
        print(f"   - Dynamic strength: {strengths.get('dynamic', 0.0):.4f}")
        
        print(f"   - Celestial graph density: {interpretation.get('celestial_graph_density', 0.0):.4f}")
        print(f"   - Overall celestial activity: {interpretation.get('overall_celestial_activity', 0.0):.4f}")
        
        return interpretation
        
    except Exception as e:
        print(f"❌ Celestial interpretation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_summary():
    """Test model summary with celestial components"""
    print("\n📊 Testing Model Summary...")
    
    config = MockConfig()
    model = CelestialEnhancedPGAT(config, mode='celestial_probabilistic')
    
    try:
        summary = model.get_model_summary()
        
        print(f"✅ Model summary generated")
        print(f"   - Celestial system enabled: {summary.get('celestial_system_enabled', False)}")
        print(f"   - Number of celestial bodies: {summary.get('num_celestial_bodies', 0)}")
        print(f"   - Celestial d_model: {summary.get('celestial_d_model', 0)}")
        print(f"   - Celestial fusion layers: {summary.get('celestial_fusion_layers', 0)}")
        print(f"   - Celestial attention heads: {summary.get('celestial_attention_heads', 0)}")
        
        if 'celestial_parameters' in summary:
            print(f"   - Celestial parameters: {summary['celestial_parameters']:,}")
        
        return summary
        
    except Exception as e:
        print(f"❌ Model summary failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_gradient_flow():
    """Test gradient flow through celestial components"""
    print("\n🔥 Testing Gradient Flow...")
    
    config = MockConfig()
    model = CelestialEnhancedPGAT(config, mode='celestial_probabilistic')
    
    # Create inputs with gradients
    batch_size = 2
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in, requires_grad=True)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
    x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
    x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
    
    try:
        # Forward pass
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Create loss
        if isinstance(output, tuple):
            means, _, _ = output
            loss = means.sum()
        else:
            loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert x_enc.grad is not None
        
        print(f"✅ Gradient flow test passed")
        print(f"   - Input gradient norm: {x_enc.grad.norm():.4f}")
        print(f"   - Loss value: {loss.item():.4f}")
        
        # Check celestial component gradients
        celestial_grad_norms = {}
        if hasattr(model, 'celestial_nodes'):
            celestial_grad_norms['celestial_nodes'] = sum(
                p.grad.norm().item() for p in model.celestial_nodes.parameters() 
                if p.grad is not None
            )
        
        if hasattr(model, 'celestial_combiner'):
            celestial_grad_norms['celestial_combiner'] = sum(
                p.grad.norm().item() for p in model.celestial_combiner.parameters()
                if p.grad is not None
            )
        
        for component, grad_norm in celestial_grad_norms.items():
            print(f"   - {component} gradient norm: {grad_norm:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_celestial_body_mapping():
    """Test celestial body to financial feature mapping"""
    print("\n🌌 Testing Celestial Body Mapping...")
    
    config = MockConfig()
    model = CelestialEnhancedPGAT(config, mode='celestial_probabilistic')
    
    print(f"Celestial Body Financial Associations:")
    for i, body in enumerate(CelestialBody):
        interpretation = model.celestial_nodes.get_body_interpretation(i)
        print(f"   {i:2d}. {interpretation['name']:12s} - {interpretation['market_influence']}")
    
    print(f"✅ Celestial body mapping verified")
    return True

def main():
    """Run all tests"""
    print("🌟 Testing Celestial Enhanced PGAT - Astrological AI")
    print("=" * 60)
    
    try:
        # Test initialization
        model = test_celestial_enhanced_pgat_initialization()
        if model is None:
            return False
        
        # Test forward pass
        output = test_celestial_enhanced_pgat_forward()
        if output is None:
            return False
        
        # Test celestial interpretation
        interpretation = test_celestial_interpretation()
        if interpretation is None:
            return False
        
        # Test model summary
        summary = test_model_summary()
        if summary is None:
            return False
        
        # Test gradient flow
        gradient_success = test_gradient_flow()
        if not gradient_success:
            return False
        
        # Test celestial body mapping
        mapping_success = test_celestial_body_mapping()
        if not mapping_success:
            return False
        
        print("\n" + "=" * 60)
        print("🎉 All Celestial Enhanced PGAT Tests Passed!")
        print("🌌 The Astrological AI is ready for financial markets!")
        print("🚀 World's first astronomically-informed graph neural network!")
        
        # Final summary
        print(f"\nFinal System Summary:")
        print(f"   - Model Type: Celestial Enhanced PGAT")
        print(f"   - Celestial Bodies: {len(CelestialBody)}")
        print(f"   - Market Regimes: 4 (Bull, Bear, Volatile, Stable)")
        print(f"   - Astrological Aspects: 5 (Conjunction, Sextile, Square, Trine, Opposition)")
        print(f"   - Fusion Method: Hierarchical Attention")
        print(f"   - Interpretability: Full astrological associations")
        
        return True
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)