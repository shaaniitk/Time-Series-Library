import torch
import torch_geometric

def main():
    print(f"PyTorch version: {torch.__version__}")
    if hasattr(torch_geometric, '__version__'):
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("GPU is available.")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        if torch.version.hip:
            print(f"ROCm version (via PyTorch): {torch.version.hip}")

        # Simple tensor operation on GPU
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device=device)
            y = torch.tensor([4.0, 5.0, 6.0], device=device)
            z = x + y
            print(f"Simple tensor addition on GPU successful: {z.cpu().numpy()}")
        except Exception as e:
            print(f"An error occurred during a simple tensor operation on the GPU: {e}")

        # torch_geometric test
        try:
            from torch_geometric.data import Data
            edge_index = torch.tensor([[0, 1, 1, 2],
                                       [1, 0, 2, 1]], dtype=torch.long, device=device)
            x_geom = torch.tensor([[-1], [0], [1]], dtype=torch.float, device=device)
            data = Data(x=x_geom, edge_index=edge_index)
            print("PyTorch Geometric data object created on GPU successfully.")
            print(data)
        except Exception as e:
            print(f"An error occurred during the PyTorch Geometric test on the GPU: {e}")

    else:
        print("GPU is not available. Please check your installation.")

if __name__ == "__main__":
    main()