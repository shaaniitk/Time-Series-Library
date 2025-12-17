import torch
import sys

def test_gpu():
    print("----------------------------------------------------------------")
    print(f"PyTorch Version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available.")
        sys.exit(1)
    
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ CUDA IS AVAILABLE! Found {device_count} device(s).")
    print(f"   Device 0: {device_name}")
    
    print("\n--- Running Tensor Addition Test ---")
    try:
        # Create tensors on GPU
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = torch.tensor([4.0, 5.0, 6.0]).cuda()
        
        print(f"   Tensor x: {x}")
        print(f"   Tensor y: {y}")
        
        # Perform addition
        z = x + y
        print(f"   Result (x + y): {z}")
        
        # Verify result
        expected = torch.tensor([5.0, 7.0, 9.0]).cuda()
        if torch.equal(z, expected):
            print("✅ Tensor addition SUCCESSFUL on GPU.")
        else:
            print("❌ Tensor addition FAILED (values do not match).")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ An error occurred during GPU operations: {e}")
        sys.exit(1)
        
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    test_gpu()
