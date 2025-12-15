import torch
import os

def safe_gpu_check():
    """Safe GPU check that works with AMD GPUs"""
    
    print("=== Safe GPU Check ===")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check basic availability
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"✓ Device count: {torch.cuda.device_count()}")
        
        # For AMD GPUs, be more careful with tensor operations
        device_name = torch.cuda.get_device_name(0)
        
        if "AMD" in device_name or "Radeon" in device_name:
            print("→ AMD GPU detected - using conservative approach")
            
            # Set environment variables that might help
            os.environ['AMD_SERIALIZE_KERNEL'] = '3'
            os.environ['HIP_VISIBLE_DEVICES'] = '0'
            
            try:
                # Try very simple operations first
                device = torch.device("cuda:0")
                
                # Start with empty tensor
                x = torch.empty(3, device=device)
                print(f"✓ Empty tensor on GPU: {x.device}")
                
                # Try filling with values
                x.fill_(1.0)
                print(f"✓ Tensor fill operation successful")
                
                # Try moving from CPU to GPU
                cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
                gpu_tensor = cpu_tensor.to(device)
                print(f"✓ CPU to GPU transfer successful: {gpu_tensor}")
                
                # Try simple math
                result = gpu_tensor + 1
                print(f"✓ GPU arithmetic successful: {result}")
                
            except Exception as e:
                print(f"✗ AMD GPU operation failed: {e}")
                print("→ Recommendation: Use CPU for now")
                return False
                
        else:
            print("→ NVIDIA GPU detected")
            # Standard NVIDIA GPU test
            try:
                device = torch.device("cuda:0")
                x = torch.tensor([1.0, 2.0, 3.0]).to(device)
                print(f"✓ NVIDIA GPU test successful: {x}")
            except Exception as e:
                print(f"✗ NVIDIA GPU test failed: {e}")
                return False
    else:
        print("✗ No GPU available - using CPU")
        return False
    
    return True

def get_recommended_device():
    """Get the recommended device for your system"""
    if safe_gpu_check():
        return torch.device("cuda:0")
    else:
        print("→ Falling back to CPU")
        return torch.device("cpu")

if __name__ == "__main__":
    print("Running safe GPU check...")
    device = get_recommended_device()
    print(f"\nRecommended device: {device}")
    
    # Test the recommended device
    print(f"\nTesting recommended device...")
    try:
        test_tensor = torch.randn(5, 5, device=device)
        result = torch.matmul(test_tensor, test_tensor.T)
        print(f"✓ Matrix multiplication test passed on {device}")
        print(f"Result shape: {result.shape}")
    except Exception as e:
        print(f"✗ Test failed on {device}: {e}")