#!/usr/bin/env python3
"""
Quick start script for seismic wave prediction
Run this to get started immediately with your seismic data
"""

import os
import subprocess
import sys

def setup_environment():
    """Setup the environment for seismic prediction"""
    print("🌊 Setting up Seismic Wave Prediction Environment...")
    
    # Create necessary directories
    os.makedirs('./dataset/seismic', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    print("✅ Directories created")

def run_seismic_training():
    """Run the seismic wave training"""
    print("🚀 Starting Seismic Wave Prediction Training...")
    
    try:
        # Run the seismic trainer
        result = subprocess.run([sys.executable, 'seismic_trainer.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print(result.stdout)
        else:
            print("❌ Training failed:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running training: {e}")

def main():
    """Main execution function"""
    print("🌊 Seismic Wave Prediction System")
    print("=" * 50)
    
    # Setup
    setup_environment()
    
    # Run training
    run_seismic_training()
    
    print("\n📊 Next Steps:")
    print("1. Replace sample data with your real seismic data")
    print("2. Adjust config parameters in seismic_wave_config.yaml")
    print("3. Modify wave processing in seismic_data_processor.py")
    print("4. Run: python seismic_trainer.py")

if __name__ == "__main__":
    main()