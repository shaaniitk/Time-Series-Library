#!/usr/bin/env python3
"""
Debug script to isolate model initialization issues
"""

import sys
import yaml
from pathlib import Path
import traceback
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.Celestial_Enhanced_PGAT import Model

class SimpleConfig:
    """Dictionary-backed configuration with attribute access support."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def test_model_initialization():
    """Test model initialization with the exact config that's failing"""
    
    # Load the config that's failing
    config_path = "results/production_workflow_tests/01_Baseline.yaml"
    
    print(f"🔍 Testing model initialization with: {config_path}")
    print("=" * 60)
    
    try:
        # Load config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        print("✅ Config loaded successfully")
        print(f"   • Model: {config_dict.get('model', 'Unknown')}")
        print(f"   • d_model: {config_dict.get('d_model', 'Unknown')}")
        print(f"   • seq_len: {config_dict.get('seq_len', 'Unknown')}")
        print(f"   • enc_in: {config_dict.get('enc_in', 'Unknown')}")
        
        # Create config object
        args = SimpleConfig(config_dict)
        
        # Add missing required attributes that might be needed
        args.task_name = 'long_term_forecast'
        args.model_name = 'Celestial_Enhanced_PGAT'
        args.data_name = 'custom'
        args.checkpoints = './checkpoints/'
        args.inverse = False
        args.cols = None
        args.num_workers = 0
        args.itr = 1
        args.train_only = False
        args.do_predict = False
        
        print("\n🚀 Attempting model initialization...")
        
        # Try to create the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   • Using device: {device}")
        
        model = Model(args).to(device)
        
        print("✅ Model initialized successfully!")
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   • Total parameters: {total_params:,}")
        print(f"   • Trainable parameters: {trainable_params:,}")
        print(f"   • Model size: ~{total_params * 4 / 1024**2:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n💥 Model initialization failed!")
        print(f"   • Error: {e}")
        print(f"   • Error type: {type(e).__name__}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_initialization()
    print(f"\n📊 Final result: {'SUCCESS' if success else 'FAILED'}")