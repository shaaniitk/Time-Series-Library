import torch
import sys

def check_gpu_comprehensive():
    """Comprehensive GPU check for both NVIDIA and AMD GPUs"""
    
    print("=== PyTorch GPU Detection Report ===")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get device count
        device_count = torch.cuda.device_count()
        print(f"Number of GPU devices: {device_count}")
        
        # Check each device
        for i in range(device_count):
            try:
                device_name = torch.cuda.get_device_name(i)
                print(f"GPU {i}: {device_name}")
                
                # Check if it's AMD or NVIDIA
                if "AMD" in device_name or "Radeon" in device_name:
                    print(f"  -> AMD GPU detected")
                    print(f"  -> ROCm/HIP support required")
                elif "NVIDIA" in device_name or "GeForce" in device_name or "Tesla" in device_name or "RTX" in device_name:
                    print(f"  -> NVIDIA GPU detected")
                    print(f"  -> CUDA support available")
                
                # Get memory info (this might fail on AMD)
                try:
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    print(f"  -> Memory allocated: {memory_allocated / 1024**2:.2f} MB")
                    print(f"  -> Memory reserved: {memory_reserved / 1024**2:.2f} MB")
                except Exception as mem_e:
                    print(f"  -> Memory info unavailable: {mem_e}")
                    
            except Exception as e:
                print(f"Error getting info for GPU {i}: {e}")
    
    # Test tensor operations
    print("\n=== Tensor Operation Tests ===")
    
    # CPU test (should always work)
    try:
        cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
        print(f"✓ CPU tensor creation successful: {cpu_tensor}")
    except Exception as e:
        print(f"✗ CPU tensor creation failed: {e}")
    
    # GPU test
    if cuda_available:
        try:
            # Try to create a simple tensor on GPU
            device = torch.device("cuda:0")
            gpu_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
            print(f"✓ GPU tensor creation successful: {gpu_tensor}")
            print(f"  -> Tensor device: {gpu_tensor.device}")
            
            # Try a simple operation
            result = gpu_tensor * 2
            print(f"✓ GPU tensor operation successful: {result}")
            
        except Exception as e:
            print(f"✗ GPU tensor operation failed: {e}")
            print("  -> This is common with AMD GPUs if ROCm is not properly configured")
            
            # Suggest fallback to CPU
            print("\n=== Fallback Recommendation ===")
            print("Consider using CPU for now:")
            print("device = torch.device('cpu')")
            print("Or check ROCm installation for AMD GPU support")
    
    # Check PyTorch compilation info
    print("\n=== PyTorch Build Info ===")
    try:
        print(f"CUDA version (compiled): {torch.version.cuda}")
    except:
        print("CUDA version: Not available")
    
    try:
        print(f"HIP version: {torch.version.hip}")
    except:
        print("HIP version: Not available")
    
    print(f"Built with CUDA: {torch.backends.cuda.is_built()}")
    print(f"Built with ROCm: {hasattr(torch.version, 'hip') and torch.version.hip is not None}")

if __name__ == "__main__":
    check_gpu_comprehensive()