# Manual Installation Guide for AMD Rembrandt GPU

Based on: https://github.com/scottt/rocm-TheRock/releases/tag/v6.5.0rc-pytorch

## Option 1: Try ROCm 6.0 PyTorch (Recommended)

```bash
# Uninstall current version
pip uninstall -y torch torchvision torchaudio

# Install ROCm 6.0 version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

## Option 2: Try ROCm 5.7 PyTorch (Fallback)

```bash
# If ROCm 6.0 doesn't work
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

## Option 3: Direct Wheel Installation (Advanced)

If the above don't work, you can try installing specific wheels from the ROCm-TheRock release:

1. Go to: https://github.com/scottt/rocm-TheRock/releases/tag/v6.5.0rc-pytorch
2. Download the appropriate `.whl` files for your Python version
3. Install with: `pip install downloaded_wheel.whl`

## Environment Variables (Required)

After installation, set these environment variables:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export AMD_SERIALIZE_KERNEL=1
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
```

## Test Your Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    x = torch.randn(3, 3, device=device)
    y = x * 2
    result = y.cpu()
    print("✅ GPU working!")
```

## Troubleshooting

If you still get the HIP error:

1. **Try different GFX versions:**
   - `HSA_OVERRIDE_GFX_VERSION=11.0.0`
   - `HSA_OVERRIDE_GFX_VERSION=10.3.0`
   - `HSA_OVERRIDE_GFX_VERSION=9.0.0`

2. **Check ROCm installation:**
   ```bash
   rocm-smi
   ```

3. **Restart terminal** after setting environment variables

4. **Add to ~/.bashrc** for permanent setup:
   ```bash
   echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
   echo 'export AMD_SERIALIZE_KERNEL=1' >> ~/.bashrc
   source ~/.bashrc
   ```