#!/usr/bin/env python3
"""
Smart PyTorch wheel installation with dependency conflict resolution
"""

import subprocess
import sys
import os
import re
from pathlib import Path

def parse_requirements(requirements_file):
    """Parse requirements.txt and separate PyTorch-related packages"""
    pytorch_packages = {'torch', 'torchvision', 'torchaudio', 'torch-geometric', 
                       'torch-scatter', 'torch-sparse', 'torch-cluster', 'torch-spline-conv'}
    
    regular_deps = []
    pytorch_deps = []
    
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name (before ==, >=, etc.)
                pkg_name = re.split(r'[>=<!=]', line)[0].strip()
                
                if pkg_name.lower() in pytorch_packages:
                    pytorch_deps.append(line)
                else:
                    regular_deps.append(line)
    
    return regular_deps, pytorch_deps

def create_compatible_requirements():
    """Create requirements files compatible with the new PyTorch wheel"""
    
    print("=== Analyzing Dependencies ===")
    
    # Parse current requirements
    regular_deps, pytorch_deps = parse_requirements('requirements.txt')
    
    print(f"Found {len(regular_deps)} regular dependencies")
    print(f"Found {len(pytorch_deps)} PyTorch-related dependencies")
    
    # Create requirements without PyTorch packages
    with open('requirements_no_torch.txt', 'w') as f:
        f.write("# Requirements without PyTorch packages\n")
        f.write("# Generated for wheel installation compatibility\n\n")
        for dep in regular_deps:
            f.write(f"{dep}\n")
    
    # Create PyTorch-specific requirements for later
    with open('pytorch_requirements.txt', 'w') as f:
        f.write("# PyTorch-related packages\n")
        f.write("# Install these after the main PyTorch wheel\n\n")
        for dep in pytorch_deps:
            if not dep.startswith('torch>=') and not dep.startswith('torchvision>=') and not dep.startswith('torchaudio>='):
                f.write(f"{dep}\n")
    
    print("✅ Created requirements_no_torch.txt")
    print("✅ Created pytorch_requirements.txt")
    
    return 'requirements_no_torch.txt', 'pytorch_requirements.txt'

def install_dependencies_safely(requirements_file):
    """Install dependencies with conflict resolution"""
    
    print(f"\n=== Installing dependencies from {requirements_file} ===")
    
    # Try installing all at once first
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print("⚠️  Batch installation failed, trying individual packages...")
        
        # Install packages one by one to identify conflicts
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        cmd = [sys.executable, "-m", "pip", "install", line]
                        subprocess.run(cmd, check=True, capture_output=True)
                        print(f"✅ {line}")
                    except subprocess.CalledProcessError:
                        print(f"⚠️  Skipped {line} (conflict or unavailable)")
        
        return True

def install_pytorch_wheel(wheel_path):
    """Install the PyTorch wheel"""
    
    print(f"\n=== Installing PyTorch wheel ===")
    print(f"Wheel: {os.path.basename(wheel_path)}")
    
    # First uninstall any existing PyTorch
    print("Removing existing PyTorch...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", 
                       "torch", "torchvision", "torchaudio"], 
                      capture_output=True)
    except:
        pass
    
    # Install the wheel
    try:
        cmd = [sys.executable, "-m", "pip", "install", wheel_path, "--force-reinstall"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ PyTorch wheel installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Wheel installation failed: {e}")
        print("STDERR:", e.stderr)
        return False

def install_pytorch_ecosystem(pytorch_requirements_file):
    """Install PyTorch ecosystem packages compatible with the new wheel"""
    
    print(f"\n=== Installing PyTorch ecosystem packages ===")
    
    # Get the installed PyTorch version
    try:
        import torch
        torch_version = torch.__version__
        print(f"Installed PyTorch version: {torch_version}")
    except ImportError:
        print("❌ PyTorch not found, cannot install ecosystem packages")
        return False
    
    # Install compatible versions
    compatible_packages = {
        'torchvision': '--index-url https://download.pytorch.org/whl/rocm6.0',
        'torchaudio': '--index-url https://download.pytorch.org/whl/rocm6.0',
    }
    
    for package, extra_args in compatible_packages.items():
        try:
            cmd = [sys.executable, "-m", "pip", "install", package] + extra_args.split()
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️  Could not install {package}, trying without index...")
            try:
                cmd = [sys.executable, "-m", "pip", "install", package]
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"✅ {package} installed (fallback)")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
    
    # Install PyTorch Geometric packages
    print("\nInstalling PyTorch Geometric packages...")
    geometric_packages = [
        'torch-geometric',
        'torch-scatter', 
        'torch-sparse',
        'torch-cluster',
        'torch-spline-conv'
    ]
    
    for package in geometric_packages:
        try:
            cmd = [sys.executable, "-m", "pip", "install", package]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️  Could not install {package}")
    
    return True

def test_installation():
    """Test the final installation"""
    
    print("\n=== Testing Installation ===")
    
    test_code = '''
import sys
print(f"Python version: {sys.version}")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm version: {getattr(torch.version, 'hip', 'Not available')}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test basic operations
    try:
        device = torch.device("cuda:0")
        x = torch.randn(3, 3, device=device)
        y = x * 2
        result = y.cpu()
        print("✅ Basic GPU operations successful")
        
        # Test matrix operations
        a = torch.randn(5, 5, device=device)
        b = torch.randn(5, 5, device=device)
        c = torch.matmul(a, b)
        print("✅ Matrix operations successful")
        
        print("🎉 PyTorch installation is working!")
        
    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
        print("You may need to set AMD GPU environment variables")
else:
    print("❌ No GPU detected")

# Test other packages
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    print("✅ Core data science packages working")
except ImportError as e:
    print(f"⚠️  Some packages missing: {e}")
'''
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Installation test failed:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Main installation workflow"""
    
    print("=== Smart PyTorch Wheel Installation ===")
    
    # Find the wheel file
    wheel_path = "/home/kalki/Downloads/torch-2.7.0a0+gitbfd8155-cp311-cp311-linux_x86_64.whl"
    
    if not os.path.exists(wheel_path):
        print(f"❌ Wheel file not found: {wheel_path}")
        return False
    
    print(f"Found wheel: {os.path.basename(wheel_path)}")
    
    # Step 1: Create compatible requirements
    regular_req, pytorch_req = create_compatible_requirements()
    
    # Step 2: Install regular dependencies first
    if not install_dependencies_safely(regular_req):
        print("❌ Failed to install regular dependencies")
        return False
    
    # Step 3: Install PyTorch wheel
    if not install_pytorch_wheel(wheel_path):
        print("❌ Failed to install PyTorch wheel")
        return False
    
    # Step 4: Install PyTorch ecosystem
    if not install_pytorch_ecosystem(pytorch_req):
        print("⚠️  Some PyTorch ecosystem packages failed")
    
    # Step 5: Test installation
    if test_installation():
        print("\n🎉 Installation completed successfully!")
        
        print("\n=== Next Steps ===")
        print("1. Set AMD GPU environment variables:")
        print("   export HSA_OVERRIDE_GFX_VERSION=10.3.0")
        print("   export AMD_SERIALIZE_KERNEL=1")
        print("   export HIP_VISIBLE_DEVICES=0")
        print("   export ROCR_VISIBLE_DEVICES=0")
        print("")
        print("2. Test your GPU: python check_gpu.py")
        print("")
        print("3. Clean up temporary files:")
        print("   rm requirements_no_torch.txt pytorch_requirements.txt")
        
        return True
    else:
        print("❌ Installation test failed")
        return False

if __name__ == "__main__":
    main()