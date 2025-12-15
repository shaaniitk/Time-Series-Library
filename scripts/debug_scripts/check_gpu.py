import torch

# 1. The most important check: is a GPU available?
if torch.cuda.is_available():
    
    # 2. How many GPUs are there?
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    # 3. What's the name of the current GPU?
    # The '0' is the index of the GPU.
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # 4. A practical test: move a tensor to the GPU
    print("\nTesting tensor allocation on GPU...")
    try:
        # Define the device
        device = torch.device("cuda:0")
        
        # Create a tensor and move it to the GPU
        x = torch.tensor([1.0, 2.0, 3.0]).to(device)
        
        print("Tensor successfully moved to GPU:")
        print(f"Tensor is on device: {x.device}")
        
        # Try to access tensor values safely (move to CPU first for AMD GPUs)
        print(f"Tensor values: {x.cpu()}")
        
        # Test a simple operation
        result = x + 1
        print(f"GPU operation result: {result.cpu()}")
        
    except Exception as e:
        print(f"An error occurred during the test: {e}")
        print("Recommendation: Use CPU device instead")
        print("device = torch.device('cpu')")

else:
    print("No GPU available. PyTorch is using the CPU.")