#!/usr/bin/env python3
"""
Install PyTorch wheel file for AMD GPU
"""

import subprocess
import sys
import os
import glob

def find_wheel_files():
    """Find all PyTorch wheel files"""
    common_locations = [
        "~/Downloads/*.whl",
        "./*.whl",
        "~/Desktop/*.whl"
    ]
    
    wheel_files = []
    for pattern in common_locations:
        expanded_pattern = os.path.expanduser(pattern)
        found_files = glob.glob(expanded_pattern)
        wheel_files.extend(found_files)
    
    # Filter for PyTorch-related wheels
    pytorch_wheels = [f for f in wheel_files if 'torch' in os.path.basename(f).lower()]
    
    return pytorch_wheels

def check_wheel_compatibility(wheel_path):
    """Check if wheel is compatible with current Python version"""
    filename = os.path.basename(wheel_path)
    current_python = f"cp{sys.version_info.major}{sys.version_info.minor}"
    
    print(f"Wheel file: {filename}")
    print(f"Current Python: {current_python}")
    
    if current_python in filename:
        print("✅ Wheel is compatible with your Python version")
        return True
    elif "cp311" in filename and current_python == "cp312":
        print("⚠️  Wheel is for Python 3.11, you have Python 3.12")
        print("   This might work but could cause issues")
        return "maybe"
    else:
        print("❌ Wheel is not compatible with your Python version")
        return False

def install_wheel(wheel_path, force=False):
    """Install the wheel file"""
    print(f"\n=== Installing {os.path.basename(wheel_path)} ===")
    
    # First, uninstall current PyTorch
    print("1. Uninstalling current PyTorch...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], 
                      check=True, capture_output=True)
        print("✅ Current PyTorch uninstalled")
    except subprocess.CalledProcessError as e:
        print("⚠️  Could not uninstall current PyTorch (might not be installed)")
    
    # Install the wheel
    print("2. Installing new PyTorch wheel...")
    cmd = [sys.executable, "-m", "pip", "install", wheel_path]
    if force:
        cmd.append("--force-reinstall")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Wheel installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print("STDERR:", e.stderr)
        return False

def test_installation():
    """Test if the installation works"""
    print("\n=== Testing Installation ===")
    
    test_code = '''
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm version: {getattr(torch.version, 'hip', 'Not available')}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        device = torch.device("cuda:0")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        print("✅ Basic GPU tensor creation successful")
        y = x * 2
        print("✅ GPU operations successful")
        result = y.cpu()
        print(f"✅ Result: {result}")
        print("🎉 Installation successful!")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        print("You may need to set environment variables")
else:
    print("❌ No GPU detected")
'''
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("Test failed:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    print("=== PyTorch Wheel Installation ===")
    
    # Find wheel files
    wheels = find_wheel_files()
    
    if not wheels:
        print("❌ No PyTorch wheel files found!")
        print("Please make sure you have downloaded the wheel file to:")
        print("- ~/Downloads/")
        print("- Current directory")
        print("- ~/Desktop/")
        return
    
    print(f"Found {len(wheels)} PyTorch wheel file(s):")
    for i, wheel in enumerate(wheels, 1):
        print(f"{i}. {wheel}")
    
    # If only one wheel, use it
    if len(wheels) == 1:
        selected_wheel = wheels[0]
        print(f"\nUsing: {selected_wheel}")
    else:
        # Let user choose
        try:
            choice = int(input(f"\nSelect wheel to install (1-{len(wheels)}): ")) - 1
            selected_wheel = wheels[choice]
        except (ValueError, IndexError):
            print("Invalid choice")
            return
    
    # Check compatibility
    compatibility = check_wheel_compatibility(selected_wheel)
    
    if compatibility == False:
        print("❌ Cannot install incompatible wheel")
        return
    elif compatibility == "maybe":
        response = input("Do you want to try installing anyway? (y/n): ")
        if response.lower() != 'y':
            print("Installation cancelled")
            return
    
    # Install the wheel
    force = compatibility == "maybe"
    if install_wheel(selected_wheel, force=force):
        # Test the installation
        test_installation()
        
        print("\n=== Next Steps ===")
        print("If GPU test failed, try setting environment variables:")
        print("export HSA_OVERRIDE_GFX_VERSION=10.3.0")
        print("export AMD_SERIALIZE_KERNEL=1")
        print("export HIP_VISIBLE_DEVICES=0")
        print("export ROCR_VISIBLE_DEVICES=0")
    else:
        print("❌ Installation failed")

if __name__ == "__main__":
    main()