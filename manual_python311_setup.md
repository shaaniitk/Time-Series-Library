# Manual Python 3.11 Setup Guide

## Step 1: Install Python 3.11 (if not already installed)

### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### Fedora/RHEL:
```bash
sudo dnf install python3.11 python3.11-devel
```

### Arch Linux:
```bash
sudo pacman -S python311
```

## Step 2: Backup Current Environment

```bash
# Export current packages
source venv/bin/activate
pip freeze > requirements_backup.txt
deactivate

# Backup current venv
mv venv venv_python312_backup
```

## Step 3: Create New Python 3.11 Environment

```bash
# Create new venv with Python 3.11
python3.11 -m venv venv

# Activate new environment
source venv/bin/activate

# Verify Python version
python --version  # Should show Python 3.11.x
```

## Step 4: Install Packages

```bash
# Upgrade pip
pip install --upgrade pip

# Install all packages except PyTorch
grep -v -E "^torch|^torchvision|^torchaudio" requirements_backup.txt > temp_requirements.txt
pip install -r temp_requirements.txt

# Install your PyTorch wheel
pip install /home/kalki/Downloads/torch-2.7.0a0+gitbfd8155-cp311-cp311-linux_x86_64.whl

# Install torchvision and torchaudio
pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

## Step 5: Set Environment Variables

```bash
# Create environment script
cat > amd_gpu_env.sh << 'EOF'
#!/bin/bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export AMD_SERIALIZE_KERNEL=1
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
EOF

chmod +x amd_gpu_env.sh
source amd_gpu_env.sh
```

## Step 6: Test Installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    x = torch.randn(3, 3, device=device)
    print('GPU test successful!')
"
```

## Rollback (if needed)

```bash
# Remove Python 3.11 environment
rm -rf venv

# Restore Python 3.12 environment
mv venv_python312_backup venv
```