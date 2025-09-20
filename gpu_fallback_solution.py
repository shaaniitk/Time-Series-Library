import torch
import os

def get_working_device():
    """
    Get a device that actually works for tensor operations.
    Tests both detection and actual usage.
    """
    print("=== Smart Device Detection ===")
    
    # First check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ No GPU detected - using CPU")
        return torch.device("cpu")
    
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    
    # Now test if we can actually USE the GPU
    device = torch.device("cuda:0")
    
    try:
        # Set environment variables that might help with AMD GPUs
        os.environ['AMD_SERIALIZE_KERNEL'] = '3'
        
        # Test 1: Create tensor on GPU
        test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
        
        # Test 2: Try to access the tensor (this is where it often fails)
        _ = test_tensor.cpu()  # Move back to CPU to test
        
        # Test 3: Try a simple operation
        result = test_tensor + 1
        
        # Test 4: Try to convert to string (this often triggers the HIP error)
        _ = str(result)
        
        print("✅ GPU operations successful - using GPU")
        return device
        
    except Exception as e:
        print(f"❌ GPU operations failed: {str(e)[:100]}...")
        print("🔄 Falling back to CPU")
        return torch.device("cpu")

def safe_tensor_operations_demo():
    """Demonstrate safe tensor operations with automatic fallback"""
    
    # Get the working device
    device = get_working_device()
    
    print(f"\n=== Using device: {device} ===")
    
    # Create tensors on the working device
    x = torch.randn(3, 3, device=device)
    y = torch.randn(3, 3, device=device)
    
    # Perform operations
    result = torch.matmul(x, y)
    
    print(f"Matrix multiplication successful!")
    print(f"Input shape: {x.shape}")
    print(f"Result shape: {result.shape}")
    print(f"Result device: {result.device}")
    
    # Safe way to display tensor values (move to CPU first if needed)
    if device.type == 'cuda':
        print(f"Sample values: {result.cpu()[:2, :2]}")
    else:
        print(f"Sample values: {result[:2, :2]}")

if __name__ == "__main__":
    safe_tensor_operations_demo()