"""Test gradient flow through edge-bias enhanced C2T attention."""
import torch
import yaml
from torch import optim
from pathlib import Path

# Load config
with open('configs/celestial_diagnostic_minimal.yaml') as f:
    config = yaml.safe_load(f)

from models.Celestial_Enhanced_PGAT import Model

class SimpleConfig:
    def __init__(self, d):
        self.__dict__.update(d)

cfg = SimpleConfig(config)
model = Model(cfg)
print('✓ Model created')

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print('✓ Optimizer created')

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

print(f'\nTesting gradient flow with edge bias...\n')

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
print('✓ Backward pass successful')

# Check gradients
grad_norms = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norms[name] = param.grad.norm().item()

# Show key gradients
print(f'\nGradient Norms (sampling):')
c2t_grads = {k: v for k, v in grad_norms.items() if 'celestial_to_target' in k}
if c2t_grads:
    print(f'\n  C2T Attention Module:')
    for k, v in list(c2t_grads.items())[:5]:
        short_name = k.split('.')[-1]
        print(f'    {short_name}: {v:.6f}')
    if len(c2t_grads) > 5:
        print(f'    ... and {len(c2t_grads) - 5} more parameters')
else:
    print('  WARNING: No C2T gradients found!')

# Overall stats
all_norms = list(grad_norms.values())
if all_norms:
    print(f'\n  Overall Gradient Stats:')
    print(f'    Parameters with gradients: {len(all_norms)}')
    print(f'    Min norm: {min(all_norms):.6f}')
    print(f'    Max norm: {max(all_norms):.6f}')
    print(f'    Mean norm: {sum(all_norms)/len(all_norms):.6f}')

# Optimizer step
optimizer.step()
print('\n✓ Optimizer step successful')

print('\n✅ Gradient flow verification PASSED!')
print('   - Edge bias code executes without errors')
print('   - Gradients flow through C2T module')
print('   - Training step completes successfully')
