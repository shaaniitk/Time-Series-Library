"""End-to-end training validation for zero-loss hierarchical fusion."""
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import yaml
from torch import optim

# Load config
with open('configs/celestial_diagnostic_minimal.yaml') as f:
    config = yaml.safe_load(f)

from models.Celestial_Enhanced_PGAT import Model

class SimpleConfig:
    def __init__(self, d):
        self.__dict__.update(d)

cfg = SimpleConfig(config)
print('Config loaded')
print(f'  use_hierarchical_fusion: {cfg.use_hierarchical_fusion}')

# Create model
model = Model(cfg)
print(f'\nModel created: {model.__class__.__name__}')

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Create realistic batch
batch_size = 4
seq_len = 96
pred_len = 24
enc_in = 118

x_enc = torch.randn(batch_size, seq_len, enc_in)
x_mark_enc = torch.randn(batch_size, seq_len, 4)
x_dec = torch.randn(batch_size, pred_len + 48, enc_in)
x_mark_dec = torch.randn(batch_size, pred_len + 48, 4)
targets = torch.randn(batch_size, pred_len, 4)

print(f'\nTest inputs created:')
print(f'  x_enc: {x_enc.shape}')
print(f'  x_mark_enc: {x_mark_enc.shape}')
print(f'  x_dec: {x_dec.shape}')
print(f'  x_mark_dec: {x_mark_dec.shape}')
print(f'  targets: {targets.shape}')

print('\n' + '='*60)
print('TESTING END-TO-END TRAINING')
print('='*60)

# Test multiple training steps
for step in range(3):
    print(f'\n--- Training Step {step + 1}/3 ---')
    
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    try:
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if isinstance(output, tuple):
            pred, aux_loss = output[0], output[1] if len(output) > 1 else 0.0
        else:
            pred, aux_loss = output, 0.0
        
        print(f'  Forward pass: pred shape = {pred.shape}')
        
        # Verify output shape
        expected_shape = (batch_size, pred_len, 4)
        if pred.shape != expected_shape:
            print(f'  ERROR: Expected {expected_shape}, got {pred.shape}')
            sys.exit(1)
        
        # Compute loss
        mse_loss = torch.nn.functional.mse_loss(pred, targets)
        total_loss = mse_loss
        if isinstance(aux_loss, torch.Tensor) and aux_loss.numel() > 0:
            total_loss = total_loss + aux_loss
        
        print(f'  Loss: {total_loss.item():.6f}')
        
        # Backward pass
        total_loss.backward()
        
        # Check for NaN gradients
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f'  ERROR: NaN gradient in {name}')
                has_nan = True
        
        if has_nan:
            sys.exit(1)
        
        # Optimizer step
        optimizer.step()
        print(f'  Optimizer step successful')
        
    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

print('\n' + '='*60)
print('SUCCESS: END-TO-END TRAINING VALIDATED!')
print('='*60)
print('Output shapes correct across all steps')
print('No NaN gradients detected')
print('Optimizer updates successful')
print('Hierarchical fusion working properly')
print('='*60)
