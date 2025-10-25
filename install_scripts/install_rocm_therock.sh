#!/bin/bash

echo "=== Installing ROCm-TheRock PyTorch for AMD Rembrandt GPU ==="
echo "Source: https://github.com/scottt/rocm-TheRock/releases/tag/v6.5.0rc-pytorch"
echo ""

# Check Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: $PYTHON_VERSION"

# Uninstall current PyTorch
echo ""
echo "1. Removing current PyTorch installation..."
pip uninstall -y torch torchvision torchaudio

# Install the compatible version
echo ""
echo "2. Installing ROCm-TheRock PyTorch..."

# Based on the GitHub release, try different installation methods
echo "Trying ROCm 6.0 index (most compatible)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Alternative: Try ROCm 5.7 if 6.0 doesn't work
if [ $? -ne 0 ]; then
    echo "ROCm 6.0 failed, trying ROCm 5.7..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
fi

# Set environment variables
echo ""
echo "3. Setting up environment variables..."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export AMD_SERIALIZE_KERNEL=1
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

echo "Set HSA_OVERRIDE_GFX_VERSION=10.3.0"
echo "Set AMD_SERIALIZE_KERNEL=1"
echo "Set HIP_VISIBLE_DEVICES=0"
echo "Set ROCR_VISIBLE_DEVICES=0"

# Create permanent environment setup
echo ""
echo "4. Creating permanent environment setup..."
cat > amd_gpu_env.sh << 'EOF'
#!/bin/bash
# AMD GPU environment variables for Rembrandt (gfx1151)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export AMD_SERIALIZE_KERNEL=1
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
echo "AMD GPU environment variables set"
EOF

chmod +x amd_gpu_env.sh

# Test installation
echo ""
echo "5. Testing installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm version: {getattr(torch.version, \"hip\", \"Not available\")}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    try:
        device = torch.device('cuda:0')
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        print('✅ Basic GPU tensor creation successful')
        y = x * 2
        print('✅ GPU operations successful')
        result = y.cpu()
        print(f'✅ GPU to CPU transfer successful: {result}')
        print('🎉 AMD GPU is working!')
    except Exception as e:
        print(f'❌ GPU test failed: {e}')
        print('Try running: source amd_gpu_env.sh')
else:
    print('❌ No GPU detected')
"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. Run: source amd_gpu_env.sh"
echo "2. Test your GPU with: python check_gpu.py"
echo "3. If it works, add the environment variables to ~/.bashrc:"
echo "   cat amd_gpu_env.sh >> ~/.bashrc"
echo ""
echo "If you still get errors, try:"
echo "- Restarting your terminal"
echo "- Running: source amd_gpu_env.sh && python your_script.py"