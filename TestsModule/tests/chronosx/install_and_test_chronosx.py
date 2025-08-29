"""
Quick ChronosX Installation and Test Script
===========================================

This script provides:
1. Installation commands for ChronosX
2. Verification of installation
3. Quick functionality test
4. Integration with modular architecture
"""

import subprocess
import sys
import torch
import numpy as np
from pathlib import Path

def install_chronosx():
    """Install ChronosX package"""
    print("TOOL Installing ChronosX package...")
    
    try:
        # Try installing chronos-forecasting
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "chronos-forecasting"
        ], capture_output=True, text=True, check=True)
        
        print("PASS ChronosX installation successful!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"FAIL Installation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def test_chronosx_installation():
    """Test if ChronosX is properly installed"""
    print("\nTEST Testing ChronosX installation...")
    
    try:
        from chronos import ChronosPipeline
        print("PASS ChronosPipeline import successful!")
        
        # Try loading a tiny model
        print(" Loading tiny ChronosX model...")
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        print("PASS Model loaded successfully!")
        
        # Test forecasting
        print("CRYSTAL Testing forecasting...")
        context = torch.randn(1, 24)  # 24 time steps
        forecast = pipeline.predict(
            context=context,
            prediction_length=12,
            num_samples=10
        )
        print(f"PASS Forecast generated! Shape: {forecast.shape}")
        print(f"CHART Forecast stats: mean={forecast.mean():.3f}, std={forecast.std():.3f}")
        
        return True, pipeline
        
    except ImportError as e:
        print(f"FAIL Import failed: {e}")
        print("IDEA Try: pip install chronos-forecasting")
        return False, None
    except Exception as e:
        print(f"WARN Error during testing: {e}")
        return False, None

def test_modular_integration():
    """Test ChronosX with modular architecture"""
    print("\n Testing modular architecture integration...")
    
    try:
        # Import modular components
        sys.path.append(str(Path(__file__).parent))
        from layers.modular.backbone.chronos_backbone import ChronosXBackbone
        from layers.modular.core.registry import unified_registry
        
        # Create registry
        print(" Creating component registry...")
        registry = unified_registry
        
        # Test ChronosX backbone creation
        print("BRAIN Creating ChronosX backbone...")
        
        # Create config object
        from configs.schemas import BackboneConfig
        
        config = BackboneConfig(
            component_name='chronos_x',
            backbone_type='chronos_x', 
            d_model=32,
            model_name='amazon/chronos-t5-tiny'
        )
        
        # Add additional parameters
        config.seq_len = 48
        config.pred_len = 12
        config.device = 'cpu'
        config.uncertainty_enabled = True
        config.model_size = 'tiny'
        
        backbone = ChronosXBackbone(config)
        print("PASS ChronosX backbone created!")
        
        # Test prediction
        print("CRYSTAL Testing backbone prediction...")
        batch_size = 2
        seq_len = 48
        d_model = 32
        
        x = torch.randn(batch_size, seq_len, d_model)
        x_mark = torch.randn(batch_size, seq_len, 4)  # temporal marks
        
        output = backbone(x, x_mark)
        print(f"PASS Prediction successful! Output shape: {output.shape}")
        
        # Test uncertainty
        if hasattr(backbone, 'get_uncertainty'):
            print("CHART Testing uncertainty quantification...")
            uncertainty = backbone.get_uncertainty()
            if uncertainty is not None:
                print(f"PASS Uncertainty available! Stats: mean={uncertainty.mean():.3f}")
            else:
                print(" No uncertainty data available yet")
        
        return True
        
    except Exception as e:
        print(f"FAIL Modular integration test failed: {e}")
        return False

def check_current_packages():
    """Check currently installed packages"""
    print(" Checking current packages...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "list"
        ], capture_output=True, text=True, check=True)
        
        lines = result.stdout.split('\n')
        relevant_packages = []
        
        for line in lines:
            if any(pkg in line.lower() for pkg in ['chronos', 'transformers', 'torch']):
                relevant_packages.append(line)
        
        print("CLIPBOARD Relevant packages:")
        for pkg in relevant_packages:
            print(f"  {pkg}")
        
        # Check specifically for chronos
        chronos_installed = any('chronos' in line.lower() for line in lines)
        return chronos_installed
        
    except Exception as e:
        print(f"FAIL Package check failed: {e}")
        return False

def main():
    """Main installation and testing workflow"""
    print("ROCKET ChronosX Installation and Test Script")
    print("=" * 50)
    
    # Step 1: Check current packages
    chronos_already_installed = check_current_packages()
    
    # Step 2: Install if needed
    if not chronos_already_installed:
        print("\nWARN ChronosX not found. Installing...")
        if not install_chronosx():
            print("FAIL Installation failed. Please try manual installation:")
            print("   pip install chronos-forecasting")
            return
    else:
        print("\nPASS ChronosX appears to be already installed!")
    
    # Step 3: Test installation
    installation_works, pipeline = test_chronosx_installation()
    
    if installation_works:
        print("\nPARTY ChronosX installation verified!")
        
        # Step 4: Test modular integration
        modular_works = test_modular_integration()
        
        if modular_works:
            print("\nTROPHY Complete Success!")
            print("PASS ChronosX installed and working")
            print("PASS Modular architecture integration working")
            print("PASS Ready for production forecasting!")
            
            # Show next steps
            print("\nCLIPBOARD Next Steps:")
            print("1. Run comprehensive tests: python test_chronos_x_simple.py")
            print("2. Try different model sizes: chronos-t5-small, chronos-t5-large")
            print("3. Test scenarios A-D with real ChronosX models")
            
        else:
            print("\nWARN ChronosX installed but modular integration has issues")
            print("IDEA Check the modular architecture components")
    
    else:
        print("\nFAIL ChronosX installation verification failed")
        print("IDEA Manual installation steps:")
        print("   1. pip install chronos-forecasting")
        print("   2. pip install --upgrade transformers")
        print("   3. Restart Python and try again")

if __name__ == "__main__":
    main()
