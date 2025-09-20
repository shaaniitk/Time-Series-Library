#!/bin/bash

echo "=== Migrating to Python 3.11 for PyTorch Wheel Compatibility ==="
echo ""

# Step 1: Check if Python 3.11 is available
echo "1. Checking Python 3.11 availability..."
if command -v python3.11 &> /dev/null; then
    echo "✅ Python 3.11 found: $(python3.11 --version)"
else
    echo "❌ Python 3.11 not found. Installing..."
    
    # Install Python 3.11 (Ubuntu/Debian)
    if command -v apt &> /dev/null; then
        echo "Installing Python 3.11 via apt..."
        sudo apt update
        sudo apt install -y python3.11 python3.11-venv python3.11-dev
    # Install Python 3.11 (Fedora/RHEL)
    elif command -v dnf &> /dev/null; then
        echo "Installing Python 3.11 via dnf..."
        sudo dnf install -y python3.11 python3.11-devel
    # Install Python 3.11 (Arch)
    elif command -v pacman &> /dev/null; then
        echo "Installing Python 3.11 via pacman..."
        sudo pacman -S python311
    else
        echo "❌ Cannot auto-install Python 3.11. Please install manually:"
        echo "   Ubuntu/Debian: sudo apt install python3.11 python3.11-venv python3.11-dev"
        echo "   Fedora: sudo dnf install python3.11 python3.11-devel"
        echo "   Arch: sudo pacman -S python311"
        exit 1
    fi
fi

# Step 2: Export current requirements
echo ""
echo "2. Exporting current requirements..."
if [ -d "venv" ]; then
    source venv/bin/activate
    pip freeze > requirements_backup.txt
    echo "✅ Current requirements saved to requirements_backup.txt"
    deactivate
else
    echo "⚠️  No existing venv found, using requirements.txt"
    cp requirements.txt requirements_backup.txt
fi

# Step 3: Backup current environment
echo ""
echo "3. Backing up current environment..."
if [ -d "venv" ]; then
    mv venv venv_python312_backup
    echo "✅ Old environment backed up as venv_python312_backup"
fi

# Step 4: Create new Python 3.11 virtual environment
echo ""
echo "4. Creating new Python 3.11 virtual environment..."
python3.11 -m venv venv
if [ $? -eq 0 ]; then
    echo "✅ New Python 3.11 virtual environment created"
else
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Step 5: Activate new environment
echo ""
echo "5. Activating new environment..."
source venv/bin/activate
echo "✅ New environment activated"
echo "Python version: $(python --version)"

# Step 6: Upgrade pip
echo ""
echo "6. Upgrading pip..."
pip install --upgrade pip

# Step 7: Install requirements (excluding PyTorch first)
echo ""
echo "7. Installing requirements (excluding PyTorch)..."
# Create a temporary requirements file without PyTorch
grep -v -E "^torch|^torchvision|^torchaudio" requirements_backup.txt > temp_requirements.txt
pip install -r temp_requirements.txt
echo "✅ Non-PyTorch packages installed"

# Step 8: Install the PyTorch wheel
echo ""
echo "8. Installing PyTorch wheel..."
TORCH_WHEEL="/home/kalki/Downloads/torch-2.7.0a0+gitbfd8155-cp311-cp311-linux_x86_64.whl"
if [ -f "$TORCH_WHEEL" ]; then
    pip install "$TORCH_WHEEL"
    if [ $? -eq 0 ]; then
        echo "✅ PyTorch wheel installed successfully!"
    else
        echo "❌ PyTorch wheel installation failed"
        exit 1
    fi
else
    echo "❌ PyTorch wheel not found at $TORCH_WHEEL"
    echo "Please check the path and try again"
    exit 1
fi

# Step 9: Install torchvision and torchaudio (if needed)
echo ""
echo "9. Installing torchvision and torchaudio..."
# Try to install compatible versions
pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Step 10: Set up AMD GPU environment variables
echo ""
echo "10. Setting up AMD GPU environment variables..."
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
source amd_gpu_env.sh

# Step 11: Test the installation
echo ""
echo "11. Testing PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'Python version: {torch.version.python}')
print(f'ROCm version: {getattr(torch.version, \"hip\", \"Not available\")}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    try:
        device = torch.device('cuda:0')
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        print('✅ GPU tensor creation successful')
        y = x * 2
        print('✅ GPU operations successful')
        result = y.cpu()
        print(f'✅ Result: {result}')
        print('🎉 AMD GPU is working with Python 3.11!')
    except Exception as e:
        print(f'❌ GPU test failed: {e}')
        print('Environment variables may need adjustment')
else:
    print('❌ No GPU detected')
"

# Cleanup
rm -f temp_requirements.txt

echo ""
echo "=== Migration Complete! ==="
echo ""
echo "Summary:"
echo "- ✅ Python 3.11 virtual environment created"
echo "- ✅ All packages reinstalled"
echo "- ✅ PyTorch wheel installed"
echo "- ✅ AMD GPU environment configured"
echo ""
echo "Next steps:"
echo "1. Always activate the environment: source venv/bin/activate"
echo "2. Set GPU variables: source amd_gpu_env.sh"
echo "3. Test your GPU: python check_gpu.py"
echo ""
echo "If you need to go back to Python 3.12:"
echo "- Remove current venv: rm -rf venv"
echo "- Restore backup: mv venv_python312_backup venv"