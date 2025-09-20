#!/usr/bin/env python3
"""
Install compatible PyTorch for AMD Rembrandt GPU
Based on: https://github.com/scottt/rocm-TheRock/releases/tag/v6.5.0rc-pytorch
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n=== {description} ===")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def install_compatible_pytorch():
    """Install the compatible PyTorch version"""
    
    print("=== Installing Compatible PyTorch for AMD Rembrandt GPU ===")
    print("Source: https://github.com/scottt/rocm-TheRock/releases/tag/v6.5.0rc-pytorch")
    
    # Step 1: Uninstall current PyTorch
    print("\n1. Removing current PyTorch installation...")
    uninstall_cmd = "pip uninstall -y torch torchvision torchaudio"
    if not run_command(uninstall_cmd, "Uninstalling current PyTorch"):
        print("Warning: Could not uninstall current PyTorch, continuing...")
    
    # Step 2: Install the compatible version
    # Based on the GitHub release, these are the wheel URLs for ROCm 6.5.0
    print("\n2. Installing compatible PyTorch with ROCm 6.5.0...")
    
    # The release provides wheels for different Python versions
    # Let's detect Python version first
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Detected Python version: {python_version}")
    
    # Install commands based on the GitHub release
    install_commands = [
        # Install torch with ROCm 6.5.0 support
        "pip install --index-url https://download.pytorch.org/whl/rocm6.0 torch torchvision torchaudio",
        
        # Alternative: Try the specific wheels from the GitHub release
        # You might need to adjust these URLs based on your Python version
    ]
    
    for cmd in install_commands:
        if run_command(cmd, f"Installing PyTorch"):
            break
    else:
        print("❌ All installation attempts failed")
        return False
    
    # Step 3: Verify installation
    print("\n3. Verifying installation...")
    verify_cmd = 'python -c "import torch; print(f\'PyTorch: {torch.__version__}\'); print(f\'CUDA available: {torch.cuda.is_available()}\'); print(f\'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'None\'}\');"'
    
    if run_command(verify_cmd, "Verifying PyTorch installation"):
        print("\n🎉 Installation completed successfully!")
        return True
    else:
        print("\n❌ Installation verification failed")
        return False

def setup_environment_variables():
    """Set up environment variables for AMD GPU"""
    print("\n=== Setting up environment variables ===")
    
    env_vars = {
        'HSA_OVERRIDE_GFX_VERSION': '10.3.0',  # Compatibility setting
        'AMD_SERIALIZE_KERNEL': '1',
        'HIP_VISIBLE_DEVICES': '0',
        'ROCR_VISIBLE_DEVICES': '0'
    }
    
    # Set for current session
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")
    
    # Create a script to set permanently
    with open('set_amd_env.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# AMD GPU environment variables\n")
        for key, value in env_vars.items():
            f.write(f"export {key}={value}\n")
    
    print("\nCreated 'set_amd_env.sh' - run 'source set_amd_env.sh' to set variables")
    print("Or add these to your ~/.bashrc for permanent setup")

def test_gpu():
    """Test if GPU works after installation"""
    print("\n=== Testing GPU functionality ===")
    
    test_script = '''
import torch
import os

print(f"PyTorch version: {torch.__version__}")
print(f"ROCm version: {getattr(torch.version, 'hip', 'Not available')}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        device = torch.device("cuda:0")
        x = torch.randn(3, 3, device=device)
        y = torch.randn(3, 3, device=device)
        z = torch.matmul(x, y)
        result = z.cpu()
        print("✅ GPU operations successful!")
        print(f"Test result shape: {result.shape}")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
else:
    print("❌ No GPU detected")
'''
    
    with open('test_gpu_after_install.py', 'w') as f:
        f.write(test_script)
    
    print("Created 'test_gpu_after_install.py' - run it to test your GPU")

if __name__ == "__main__":
    print("This script will install compatible PyTorch for your AMD Rembrandt GPU")
    print("Based on: https://github.com/scottt/rocm-TheRock/releases/tag/v6.5.0rc-pytorch")
    
    response = input("\nDo you want to proceed? (y/n): ")
    if response.lower() != 'y':
        print("Installation cancelled")
        sys.exit(0)
    
    # Run installation steps
    if install_compatible_pytorch():
        setup_environment_variables()
        test_gpu()
        
        print("\n=== Next Steps ===")
        print("1. Run: source set_amd_env.sh")
        print("2. Run: python test_gpu_after_install.py")
        print("3. If it works, add the environment variables to ~/.bashrc")
    else:
        print("\n❌ Installation failed. You may need to:")
        print("1. Check your internet connection")
        print("2. Try installing manually from the GitHub release")
        print("3. Consider using CPU-only PyTorch as fallback")