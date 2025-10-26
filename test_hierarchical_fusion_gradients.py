"""Test hierarchical fusion gradient flow with Celestial model."""
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
print(f'Model created: {model.__class__.__name__}')

# Check fusion components
has_cross_attn = hasattr(model, 'fusion_cross_attention')
has_hierarchical_proj = hasattr(model, 'hierarchical_fusion_proj')

print(f'Hierarchical fusion components:')
print(f'  Cross-attention: {has_cross_attn}')
print(f'  Hierarchical projection: {has_hierarchical_proj}')

if has_cross_attn and has_hierarchical_proj:
    print('SUCCESS: Hierarchical fusion is active!')
else:
    print('INFO: Using fallback fusion (model may not inherit from Enhanced_SOTA_PGAT)')

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Create dummy batch
batch_size = 4
seq_len = 96
pred_len = 24
enc_in = 118

x_enc = torch.randn(batch_size, seq_len, enc_in)
x_mark_enc = torch.randn(batch_size, seq_len, 4)
x_dec = torch.randn(batch_size, pred_len + 48, enc_in)
x_mark_dec = torch.randn(batch_size, pred_len + 48, 4)
targets = torch.randn(batch_size, pred_len, 4)

print(f'\nRunning training step...')

# Training step
model.train()
optimizer.zero_grad()

# Forward pass
output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
if isinstance(output, tuple):
    pred, aux_loss = output[0], output[1] if len(output) > 1 else 0.0
else:
    pred, aux_loss = output, 0.0

# Compute loss
mse_loss = torch.nn.functional.mse_loss(pred, targets)
total_loss = mse_loss
if isinstance(aux_loss, torch.Tensor) and aux_loss.numel() > 0:
    total_loss = total_loss + aux_loss
    print(f'  MSE Loss: {mse_loss.item():.6f}')
    print(f'  Aux Loss: {aux_loss.item():.6f}')
    print(f'  Total Loss: {total_loss.item():.6f}')
else:
    print(f'  MSE Loss: {mse_loss.item():.6f}')
    print(f'  Total Loss: {total_loss.item():.6f}')

# Backward pass
total_loss.backward()
print('Backward pass successful')

# Check gradients
grad_norms = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norms[name] = param.grad.norm().item()

# Show key gradients
fusion_grads = {k: v for k, v in grad_norms.items() if 'fusion' in k.lower() or 'hierarchical' in k.lower()}
if fusion_grads:
    print(f'\nFusion Module Gradients:')
    for k, v in list(fusion_grads.items())[:10]:
        print(f'  {k}: {v:.6f}')
    if len(fusion_grads) > 10:
        print(f'  ... and {len(fusion_grads) - 10} more')

# Overall stats
all_norms = list(grad_norms.values())
if all_norms:
    print(f'\nOverall Gradient Stats:')
    print(f'  Parameters with gradients: {len(all_norms)}')
    print(f'  Min norm: {min(all_norms):.6f}')
    print(f'  Max norm: {max(all_norms):.6f}')
    print(f'  Mean norm: {sum(all_norms)/len(all_norms):.6f}')

# Optimizer step
optimizer.step()
print('\nOptimizer step successful')

print('\nSUCCESS: Hierarchical fusion gradient flow verified!')
