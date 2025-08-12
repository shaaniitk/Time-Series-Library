import torch

# Test individual loss functions
def test_mape():
    # MAPE Loss implementation
    class MAPELoss:
        def __init__(self):
            self.output_dim_multiplier = 1
        
        def forward(self, predictions, targets):
            epsilon = 1e-8
            targets_safe = torch.clamp(torch.abs(targets), min=epsilon)
            mape = torch.mean(torch.abs((targets - predictions) / targets_safe)) * 100
            return mape
        
        def __call__(self, predictions, targets):
            return self.forward(predictions, targets)
    
    # Test
    pred = torch.randn(2, 10, 3)
    target = torch.abs(torch.randn(2, 10, 3)) + 0.1
    
    mape = MAPELoss()
    loss = mape(pred, target)
    print(f"MAPE Loss: {loss.item():.4f}")
    return True

test_mape()
print("Basic test completed successfully")
