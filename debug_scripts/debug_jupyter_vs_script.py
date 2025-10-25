import torch
import os
import sys

print("=== Environment Debug ===")
print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check environment variables that might affect GPU behavior
gpu_env_vars = [
    'CUDA_VISIBLE_DEVICES',
    'HIP_VISIBLE_DEVICES', 
    'AMD_SERIALIZE_KERNEL',
    'ROCR_VISIBLE_DEVICES',
    'GPU_DEVICE_ORDINAL',
    'PYTORCH_CUDA_ALLOC_CONF'
]

print("\n=== GPU Environment Variables ===")
for var in gpu_env_vars:
    value = os.environ.get(var, 'Not set')
    print(f"{var}: {value}")

# Test the exact same operations as your Jupyter notebook
print("\n=== GPU Test (same as Jupyter) ===")
if torch.cuda.is_available():
    try:
        device = torch.device("cuda:0")
        print(f"Device created: {device}")
        
        # Create tensor on GPU
        gpu_tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
        print(f"Tensor created on GPU: {gpu_tensor.device}")
        
        # This is where it might fail - try to print the tensor
        print("Attempting to print GPU tensor...")
        print(f"GPU tensor: {gpu_tensor}")
        
        print("✅ SUCCESS: GPU operations completed!")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print(f"Error type: {type(e).__name__}")