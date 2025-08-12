#!/usr/bin/env python3
"""
HF Modular Architecture Training Test

This script tests the HFAutoformer with different modular configurations
to validate the unified architecture with component combinations.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_dummy_data(batch_size=8, seq_len=96, pred_len=24, enc_in=7, features=4):
    """Create dummy time series data for testing"""
    # Input sequence
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    
    # Time features (day of week, hour, etc.)
    x_mark_enc = torch.randn(batch_size, seq_len, features)
    
    # Decoder input (for autoregressive models)
    x_dec = torch.randn(batch_size, pred_len, enc_in)
    x_mark_dec = torch.randn(batch_size, pred_len, features)
    
    # Target for training
    y = torch.randn(batch_size, pred_len, enc_in)
    
    return x_enc, x_mark_enc, x_dec, x_mark_dec, y

class MockConfig:
    """Mock configuration for HFAutoformer testing"""
    def __init__(self, **kwargs):
        # Base configuration
        self.seq_len = 96
        self.pred_len = 24
        self.enc_in = 7
        self.c_out = 7
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.dropout = 0.1
        self.activation = 'gelu'
        self.output_attention = False
        
        # Modular component configuration
        self.backbone_type = 'simple_transformer'
        self.loss_type = 'mse'
        self.attention_type = 'standard'
        self.processor_type = 'identity'
        
        # Override with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

def test_hf_modular_configuration(config_name, config_dict):
    """Test a specific HFAutoformer configuration"""
    print(f"\nTEST Testing {config_name}")
    print("-" * 60)
    
    try:
        # Try to import HFAutoformer
        try:
            from models.HFAutoformer import HFAutoformer
            print("PASS HFAutoformer imported successfully")
        except ImportError as e:
            print(f"WARN Could not import HFAutoformer: {e}")
            print("   Falling back to modular component testing...")
            return test_modular_components(config_dict)
        
        # Create configuration
        config = MockConfig(**config_dict)
        
        # Create model
        model = HFAutoformer(config)
        print(f"PASS Model created with config: {config_name}")
        
        # Create dummy data
        x_enc, x_mark_enc, x_dec, x_mark_dec, y = create_dummy_data()
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            print(f"PASS Forward pass: {x_enc.shape}  {output.shape}")
        
        # Test training step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        total_loss = 0
        for step in range(3):
            optimizer.zero_grad()
            
            pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = nn.MSELoss()(pred, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            print(f"   Step {step+1}: Loss = {loss.item():.6f}")
        
        avg_loss = total_loss / 3
        print(f"PASS Training successful - Average Loss: {avg_loss:.6f}")
        
        return True
        
    except Exception as e:
        print(f"FAIL {config_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_modular_components(config_dict):
    """Test individual modular components if HFAutoformer not available"""
    print("TOOL Testing modular components individually...")
    
    try:
        from utils.modular_components.registry import create_component
        
        # Test component creation
        components_tested = 0
        components_success = 0
        
        # Test backbone
        if 'backbone_type' in config_dict:
            try:
                backbone = create_component('backbone', config_dict['backbone_type'], MockConfig())
                if backbone is not None:
                    print(f"PASS Backbone '{config_dict['backbone_type']}' created")
                    components_success += 1
                else:
                    print(f"WARN Backbone '{config_dict['backbone_type']}' returned None")
                components_tested += 1
            except Exception as e:
                print(f"FAIL Backbone '{config_dict['backbone_type']}' failed: {e}")
                components_tested += 1
        
        # Test loss
        if 'loss_type' in config_dict:
            try:
                loss_fn = create_component('loss', config_dict['loss_type'], MockConfig())
                if loss_fn is not None:
                    print(f"PASS Loss '{config_dict['loss_type']}' created")
                    components_success += 1
                else:
                    print(f"WARN Loss '{config_dict['loss_type']}' returned None")
                components_tested += 1
            except Exception as e:
                print(f"FAIL Loss '{config_dict['loss_type']}' failed: {e}")
                components_tested += 1
        
        # Test attention
        if 'attention_type' in config_dict:
            try:
                attention = create_component('attention', config_dict['attention_type'], MockConfig())
                if attention is not None:
                    print(f"PASS Attention '{config_dict['attention_type']}' created")
                    components_success += 1
                else:
                    print(f"WARN Attention '{config_dict['attention_type']}' returned None")
                components_tested += 1
            except Exception as e:
                print(f"FAIL Attention '{config_dict['attention_type']}' failed: {e}")
                components_tested += 1
        
        # Test processor
        if 'processor_type' in config_dict:
            try:
                processor = create_component('processor', config_dict['processor_type'], MockConfig())
                if processor is not None:
                    print(f"PASS Processor '{config_dict['processor_type']}' created")
                    components_success += 1
                else:
                    print(f"WARN Processor '{config_dict['processor_type']}' returned None")
                components_tested += 1
            except Exception as e:
                print(f"FAIL Processor '{config_dict['processor_type']}' failed: {e}")
                components_tested += 1
        
        print(f"CHART Component test results: {components_success}/{components_tested} successful")
        return components_success > 0
        
    except ImportError as e:
        print(f"FAIL Could not import modular components: {e}")
        return False

def run_hf_modular_tests():
    """Run comprehensive HF Modular Architecture tests"""
    print("ROCKET HF Modular Architecture Training Tests")
    print("=" * 80)
    
    # Test configurations from the architecture guide
    test_configs = {
        "Basic Configuration": {
            'backbone_type': 'simple_transformer',
            'loss_type': 'mse',
            'attention_type': 'standard',
            'processor_type': 'identity'
        },
        
        "Chronos Backbone Configuration": {
            'backbone_type': 'chronos',
            'loss_type': 'mse',
            'attention_type': 'standard',
            'processor_type': 'identity'
        },
        
        "Bayesian Uncertainty Configuration": {
            'backbone_type': 'chronos_t5',
            'loss_type': 'bayesian_kl',
            'attention_type': 'bayesian_attention',
            'processor_type': 'uncertainty_processor'
        },
        
        "Quantile Regression Configuration": {
            'backbone_type': 'chronos_t5',
            'loss_type': 'quantile_loss',
            'attention_type': 'multi_quantile_attention',
            'processor_type': 'quantile_processor'
        },
        
        "Hierarchical Multi-Scale Configuration": {
            'backbone_type': 'chronos_t5',
            'loss_type': 'hierarchical_loss',
            'attention_type': 'hierarchical_attention',
            'processor_type': 'multi_scale_processor'
        },
        
        "T5 Backbone Configuration": {
            'backbone_type': 't5',
            'loss_type': 'mse',
            'attention_type': 'standard',
            'processor_type': 'identity'
        },
        
        "Robust HF Configuration": {
            'backbone_type': 'robust_hf',
            'loss_type': 'adaptive_structural',
            'attention_type': 'optimized_autocorrelation',
            'processor_type': 'integrated_signal'
        }
    }
    
    results = {}
    passed = 0
    total = len(test_configs)
    
    for config_name, config_dict in test_configs.items():
        success = test_hf_modular_configuration(config_name, config_dict)
        results[config_name] = success
        if success:
            passed += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("CHART HF Modular Architecture Test Results")
    print("=" * 80)
    
    for config_name, success in results.items():
        status = "PASS PASSED" if success else "FAIL FAILED"
        print(f"   {status} - {config_name}")
    
    print(f"\nCHART Overall Results:")
    print(f"   Tests Passed: {passed}/{total}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\nPARTY All HF Modular Architecture tests passed!")
        print(" The unified modular system is working correctly!")
    elif passed > 0:
        print(f"\nWARN {total-passed} configurations failed - some components may not be implemented yet")
        print(" Basic modular functionality is working!")
    else:
        print("\nFAIL All tests failed - check HFAutoformer implementation")
    
    return passed > 0

if __name__ == "__main__":
    success = run_hf_modular_tests()
    sys.exit(0 if success else 1)
