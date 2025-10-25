import os
import torch

# Set AMD/ROCm environment variables BEFORE importing torch operations
os.environ['AMD_SERIALIZE_KERNEL'] = '3'
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['ROCR_VISIBLE_DEVICES'] = '0'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'  # Try this for compatibility

print("=== AMD GPU Fix Attempt ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        device = torch.device("cuda:0")
        
        # Test basic tensor creation
        print("Testing tensor creation...")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        print(f"✅ Tensor created: device={x.device}")
        
        # Test tensor operations
        print("Testing tensor operations...")
        y = x * 2
        print(f"✅ Multiplication successful")
        
        # Test matrix operations
        print("Testing matrix operations...")
        a = torch.randn(3, 3, device=device)
        b = torch.randn(3, 3, device=device)
        c = torch.matmul(a, b)
        print(f"✅ Matrix multiplication successful")
        
        # Test moving to CPU (this often triggers the error)
        print("Testing GPU to CPU transfer...")
        cpu_result = c.cpu()
        print(f"✅ GPU to CPU transfer successful")
        
        print("🎉 ALL TESTS PASSED - AMD GPU is working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Trying additional fixes...")
        
        # Additional environment variables to try
        additional_vars = {
            'HSA_OVERRIDE_GFX_VERSION': '11.0.0',
            'HIP_FORCE_DEV_KERNARG': '1',
            'AMD_LOG_LEVEL': '1'
        }
        
        for var, value in additional_vars.items():
            os.environ[var] = value
            print(f"Set {var}={value}")