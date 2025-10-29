#!/usr/bin/env python3
"""
Test script to verify the production configuration initializes correctly
"""

import sys
import os
import yaml
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_loading():
    """Test loading the production configuration"""
    print("Testing Production Configuration Loading...")
    
    config_path = "configs/celestial_production_deep_ultimate.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Configuration loaded successfully from {config_path}")
        
        # Check key parameters
        key_params = {
            'model': config.get('model'),
            'd_model': config.get('d_model'),
            'n_heads': config.get('n_heads'),
            'seq_len': config.get('seq_len'),
            'pred_len': config.get('pred_len'),
            'num_celestial_bodies': config.get('num_celestial_bodies'),
            'celestial_dim': config.get('celestial_dim'),
            'enable_mdn_decoder': config.get('enable_mdn_decoder'),
        }
        
        print("\nKey Configuration Parameters:")
        for key, value in key_params.items():
            print(f"  {key}: {value}")
        
        # Verify dimension compatibility
        d_model = config.get('d_model', 0)
        num_celestial_bodies = config.get('num_celestial_bodies', 1)
        
        if d_model % num_celestial_bodies == 0:
            print(f"✅ Dimension compatibility: d_model ({d_model}) is divisible by num_celestial_bodies ({num_celestial_bodies})")
        else:
            print(f"❌ Dimension compatibility: d_model ({d_model}) is NOT divisible by num_celestial_bodies ({num_celestial_bodies})")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_model_args_creation():
    """Test creating model arguments from config"""
    print("\nTesting Model Arguments Creation...")
    
    try:
        # Import the args creation function
        from scripts.train.train_celestial_production import create_args_from_config
        
        config_path = "configs/celestial_production_deep_ultimate.yaml"
        args = create_args_from_config(config_path)
        
        print(f"✅ Model arguments created successfully")
        print(f"  Model: {args.model}")
        print(f"  d_model: {args.d_model}")
        print(f"  seq_len: {args.seq_len}")
        print(f"  pred_len: {args.pred_len}")
        print(f"  Device: {args.use_gpu}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model arguments creation failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("PRODUCTION CONFIGURATION INITIALIZATION TEST")
    print("=" * 60)
    
    success = True
    
    # Test configuration loading
    success &= test_config_loading()
    
    # Test model arguments creation
    success &= test_model_args_creation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 PRODUCTION CONFIGURATION IS READY!")
        print("✅ All initialization tests passed")
        print("✅ Dimensions are compatible")
        print("✅ Configuration is valid")
        print("\n🚀 Ready to run:")
        print("python scripts/train/train_celestial_production.py --config configs/celestial_production_deep_ultimate.yaml")
    else:
        print("❌ CONFIGURATION HAS ISSUES!")
        print("Please fix the configuration before running production training")
    print("=" * 60)