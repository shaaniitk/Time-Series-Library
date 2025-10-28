#!/usr/bin/env python3
"""
Debug script to catch the exact tuple index error
"""
import traceback
import torch
from Enhanced_SOTA_PGAT_Refactored import Enhanced_SOTA_PGAT
from component_configuration_manager import ComponentConfigurationManager

def debug_tuple_error():
    """Debug the tuple index out of range error"""
    print("🔍 DEBUGGING TUPLE INDEX ERROR")
    print("=" * 50)
    
    try:
        # Initialize configuration manager
        config_manager = ComponentConfigurationManager()
        
        # Get baseline configuration
        config = config_manager.get_configuration("01_Baseline")
        print(f"✅ Configuration loaded: {config['name']}")
        
        # Initialize model
        model = Enhanced_SOTA_PGAT(config)
        print("✅ Model initialized successfully")
        
        # Create dummy data to test forward pass
        batch_size = 2
        seq_len = 96
        n_features = 113
        
        print(f"📊 Creating test batch: {batch_size}x{seq_len}x{n_features}")
        
        # Create test input
        x = torch.randn(batch_size, seq_len, n_features)
        print(f"✅ Input tensor created: {x.shape}")
        
        # Try forward pass
        print("🚀 Attempting forward pass...")
        with torch.no_grad():
            output = model(x)
        
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
        
    except Exception as e:
        print(f"💥 ERROR CAUGHT: {type(e).__name__}: {e}")
        print("\n🔍 FULL TRACEBACK:")
        print("-" * 50)
        traceback.print_exc()
        print("-" * 50)
        
        # Try to identify the exact line
        tb = traceback.extract_tb(e.__traceback__)
        print(f"\n📍 ERROR LOCATION:")
        for frame in tb:
            print(f"   File: {frame.filename}")
            print(f"   Line: {frame.lineno}")
            print(f"   Function: {frame.name}")
            print(f"   Code: {frame.line}")
            print()

if __name__ == "__main__":
    debug_tuple_error()