#!/usr/bin/env python3
"""
Simple smart wheel installation with dependency handling
"""

import subprocess
import sys
import os

def install_wheel_with_deps():
    """Install PyTorch wheel and handle dependency conflicts"""
    
    wheel_path = "/home/kalki/Downloads/torch-2.7.0a0+gitbfd8155-cp311-cp311-linux_x86_64.whl"
    
    print("=== Smart Wheel Installation ===")
    print(f"Installing: {os.path.basename(wheel_path)}")
    
    if not os.path.exists(wheel_path):
        print(f"❌ Wheel not found: {wheel_path}")
        return False
    
    # Step 1: Uninstall existing PyTorch to avoid conflicts
    print("\n1. Removing existing PyTorch...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "-y",
            "torch", "torchvision", "torchaudio", 
            "torch-geometric", "torch-scatter", "torch-sparse", 
            "torch-cluster", "torch-spline-conv"
        ], capture_output=True)
        print("✅ Existing PyTorch packages removed")
    except:
        print("⚠️  No existing PyTorch found (or removal failed)")
    
    # Step 2: Install the wheel with dependency resolution
    print("\n2. Installing PyTorch wheel...")
    try:
        # Use --force-reinstall and --no-deps to avoid conflicts
        cmd = [
            sys.executable, "-m", "pip", "install", 
            wheel_path, 
            "--force-reinstall",
            "--no-deps"  # Install without dependencies first
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print("✅ PyTorch wheel installed (no deps)")
        
        # Now install missing dependencies
        print("Installing PyTorch dependencies...")
        deps_cmd = [
            sys.executable, "-m", "pip", "install",
            "numpy", "typing-extensions", "sympy", "networkx", 
            "jinja2", "fsspec", "filelock"
        ]
        subprocess.run(deps_cmd, check=True, capture_output=True)
        print("✅ PyTorch dependencies installed")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Wheel installation failed: {e}")
        return False
    
    # Step 3: Install compatible torchvision and torchaudio
    print("\n3. Installing torchvision and torchaudio...")
    try:
        # Try ROCm compatible versions
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/rocm6.0"
        ], check=True, capture_output=True)
        print("✅ torchvision and torchaudio installed")
    except:
        print("⚠️  ROCm versions failed, trying default...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torchvision", "torchaudio"
            ], check=True, capture_output=True)
            print("✅ torchvision and torchaudio installed (default)")
        except:
            print("❌ Could not install torchvision/torchaudio")
    
    # Step 4: Reinstall PyTorch Geometric packages
    print("\n4. Installing PyTorch Geometric...")
    geometric_packages = [
        "torch-geometric",
        "torch-scatter", 
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv"
    ]
    
    for package in geometric_packages:
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True)
            print(f"✅ {package}")
        except:
            print(f"⚠️  Could not install {package}")
    
    # Step 5: Test installation
    print("\n5. Testing installation...")
    try:
        test_result = subprocess.run([
            sys.executable, "-c", 
            "import torch; print(f'PyTorch {torch.__version__} installed'); print(f'CUDA: {torch.cuda.is_available()}')"
        ], check=True, capture_output=True, text=True)
        
        print(test_result.stdout.strip())
        print("✅ Installation successful!")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation test failed: {e}")
        return False

if __name__ == "__main__":
    if install_wheel_with_deps():
        print("\n🎉 PyTorch wheel installed successfully!")
        print("\nNext steps:")
        print("1. Set AMD GPU environment variables:")
        print("   export HSA_OVERRIDE_GFX_VERSION=10.3.0")
        print("   export AMD_SERIALIZE_KERNEL=1")
        print("2. Test GPU: python check_gpu.py")
    else:
        print("\n❌ Installation failed")
        print("You may need to manually resolve dependency conflicts")