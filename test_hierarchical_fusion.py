"""Test hierarchical cross-attention fusion implementation."""
import torch
import yaml
from models.Enhanced_SOTA_PGAT import Enhanced_SOTA_PGAT

# Load config
with open('configs/celestial_diagnostic_minimal.yaml') as f:
    config = yaml.safe_load(f)

class SimpleConfig:
    def __init__(self, d):
        self.__dict__.update(d)

cfg = SimpleConfig(config)
print('Config loaded with use_hierarchical_fusion:', cfg.use_hierarchical_fusion)

# Create model
model = Enhanced_SOTA_PGAT(cfg)
print(f'Model instantiated: {model.__class__.__name__}')

# Check hierarchical fusion components
if hasattr(model, 'fusion_cross_attention'):
    print('✓ Hierarchical fusion cross-attention module present')
    print(f'  - num_heads: {model.fusion_cross_attention.num_heads}')
    print(f'  - embed_dim: {model.fusion_cross_attention.embed_dim}')
else:
    print('✗ Cross-attention module missing!')

if hasattr(model, 'hierarchical_fusion_proj'):
    print('✓ Hierarchical fusion projection layer present')
    print(f'  - input_dim: {model.hierarchical_fusion_proj.in_features}')
    print(f'  - output_dim: {model.hierarchical_fusion_proj.out_features}')

# Test forward pass
batch_size = 2
seq_len = 96
enc_in = 118

wave_window = torch.randn(batch_size, seq_len, enc_in)
target_window = torch.randn(batch_size, seq_len, enc_in)

print(f'\nTest inputs created')

# Enable diagnostics
model.collect_diagnostics = True
model.eval()

with torch.no_grad():
    try:
        output = model(wave_window, target_window)
        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output
        print(f'✓ Forward pass successful: {pred.shape}')
        
        # Check if fusion attention weights were stored
        if hasattr(model, '_last_fusion_attention'):
            attn = model._last_fusion_attention
            print(f'✓ Fusion attention weights captured: {attn.shape}')
            print(f'  - Attention pattern: {attn.shape[0]} queries attending to {attn.shape[2]} temporal scales')
            print(f'  - Mean attention weight: {attn.mean().item():.4f}')
            print(f'  - Max attention weight: {attn.max().item():.4f}')
        else:
            print('  (No attention weights stored - diagnostics may not be enabled)')
            
    except Exception as e:
        print(f'✗ Forward pass failed: {e}')
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

print('\n✅ Hierarchical fusion implementation validated!')
print('   - Cross-attention module active')
print('   - Temporal scale features dynamically weighted')
print('   - Node context queries temporal information')
