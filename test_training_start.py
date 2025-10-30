#!/usr/bin/env python3
"""
Test script to isolate the training start issue
"""

import sys
import os
import traceback

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_training_start():
    """Test the training start process"""
    print("Testing training start process...")
    
    try:
        # Import the training function
        from scripts.train.train_celestial_production import train_celestial_pgat_production
        
        # Set up arguments
        config_path = "configs/celestial_production_deep_ultimate.yaml"
        
        print(f"Starting training with config: {config_path}")
        
        # Call the training function
        train_celestial_pgat_production(config_path)
        
        print("✅ Training completed successfully")
        return True
        
    except KeyboardInterrupt:
        print("⚠️  Training interrupted by user")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING START TEST")
    print("=" * 60)
    
    success = test_training_start()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TRAINING START TEST COMPLETED!")
    else:
        print("❌ TRAINING START TEST FAILED!")
    print("=" * 60)