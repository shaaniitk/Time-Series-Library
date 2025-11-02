"""
Verify that gradients are flowing through spatiotemporal encoder by loading
the checkpoint and running a quick forward+backward pass.
"""

import torch
import yaml
from models.Celestial_Enhanced_PGAT_Modular import Model


def verify_gradients() -> None:
    """Verify gradient flow through spatiotemporal encoder."""
    checkpoint_path = "checkpoints/celestial_enhanced_pgat_production_overnight/best_model.pth"
    
    print(f"Loading config...")
    with open("configs/celestial_enhanced_pgat_production.yaml") as f:
        config = yaml.safe_load(f)
    
    args = type('Args', (), {})()
    for key, value in config.items():
        setattr(args, key, value)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model = Model(args)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    print("\n1. CHECKING ENCODER_BLEND_GATE VALUE:")
    blend_value = model.encoder_module.encoder_blend_gate.item()
    blend_weight = torch.sigmoid(torch.tensor(blend_value)).item()
    print(f"   Raw value: {blend_value:.4f}")
    print(f"   Sigmoid(value): {blend_weight:.4f}")
    print(f"   Petri weight: {blend_weight:.2%}")
    print(f"   Spatiotemporal weight: {(1-blend_weight):.2%}")
    
    # Create dummy inputs
    batch_size = 4
    time_feat_dim = 4  # Default time features (hour, day, weekday, month)
    x = torch.randn(batch_size, args.seq_len, args.enc_in)
    x_mark_enc = torch.randn(batch_size, args.seq_len, time_feat_dim)
    x_dec = torch.randn(batch_size, args.pred_len, args.dec_in)
    x_mark_dec = torch.randn(batch_size, args.pred_len, time_feat_dim)
    
    print("\n2. RUNNING FORWARD+BACKWARD PASS:")
    outputs = model(x, x_mark_enc, x_dec, x_mark_dec)
    loss = outputs.mean()
    loss.backward()
    print(f"   ✅ Forward+backward completed, loss={loss.item():.6f}")
    
    print("\n3. CHECKING SPATIOTEMPORAL ENCODER GRADIENTS:")
    zero_grads = []
    nonzero_grads = []
    
    for name, param in model.encoder_module.spatiotemporal_encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-8:
                nonzero_grads.append((name, grad_norm))
            else:
                zero_grads.append(name)
    
    if nonzero_grads:
        print(f"   ✅ {len(nonzero_grads)} parameters with NON-ZERO gradients:")
        for name, grad_norm in sorted(nonzero_grads, key=lambda x: -x[1])[:10]:
            print(f"      {name}: {grad_norm:.8f}")
        if len(nonzero_grads) > 10:
            print(f"      ... and {len(nonzero_grads) - 10} more")
    
    if zero_grads:
        print(f"   ⚠️  {len(zero_grads)} parameters with ZERO gradients:")
        for name in zero_grads[:3]:
            print(f"      {name}")
    
    print("\n4. CHECKING ENCODER_BLEND_GATE GRADIENT:")
    blend_grad = model.encoder_module.encoder_blend_gate.grad
    if blend_grad is not None:
        grad_norm = blend_grad.norm().item()
        print(f"   ✅ Gradient norm: {grad_norm:.8f}")
    else:
        print(f"   ❌ No gradient (should not happen)")
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS:")
    print("=" * 70)
    if nonzero_grads:
        print(f"✅ Spatiotemporal encoder: {len(nonzero_grads)} params with gradients")
        print(f"   Fix SUCCESSFUL - gradient starvation resolved!")
    else:
        print(f"❌ Spatiotemporal encoder: All gradients are zero")
        print(f"   Fix FAILED - gradient starvation persists")


if __name__ == "__main__":
    verify_gradients()
