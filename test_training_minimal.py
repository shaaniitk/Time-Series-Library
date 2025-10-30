#!/usr/bin/env python3
"""
Minimal test to run just a few training batches
"""

import sys
import os
import traceback
import signal

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n⚠️  Training interrupted by user")
    sys.exit(0)

def test_minimal_training():
    """Test running just a few training batches"""
    print("Testing minimal training (will interrupt after a few batches)...")
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Import the training function
        from scripts.train.train_celestial_production import train_celestial_pgat_production
        
        # Set up arguments
        config_path = "configs/celestial_production_fixed.yaml"
        
        print(f"Starting minimal training with config: {config_path}")
        print("Press Ctrl+C after a few batches to stop...")
        
        # Call the training function
        train_celestial_pgat_production(config_path)
        
        print("✅ Training completed successfully")
        return True
        
    except KeyboardInterrupt:
        print("⚠️  Training interrupted by user - this is expected")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MINIMAL TRAINING TEST")
    print("=" * 60)
    
    success = test_minimal_training()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 MINIMAL TRAINING TEST COMPLETED!")
    else:
        print("❌ MINIMAL TRAINING TEST FAILED!")
    print("=" * 60)