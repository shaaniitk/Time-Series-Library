"""
Verification script to check if encoder soft blending fix is working.
Checks:
1. Is encoder_blend_gate parameter created?
2. Do spatiotemporal encoder parameters have non-zero gradients?
"""

import torch
import yaml
from models.Celestial_Enhanced_PGAT_Modular import Model


def verify_encoder_fix() -> None:
    """Verify the encoder fix by checking parameter creation and gradients."""
    print("Loading config...")
    with open("configs/celestial_enhanced_pgat_production.yaml") as f:
        config = yaml.safe_load(f)
    
    args = type('Args', (), {})()
    for key, value in config.items():
        setattr(args, key, value)
    
    # Override for quick test
    args.seq_len = 30
    args.pred_len = 10
    args.enc_in = 113
    
    print("Initializing model...")
    model = Model(args)
    
    # Check if blend_gate exists (it's created lazily during forward)
    print("\n1. CHECKING ENCODER_BLEND_GATE PARAMETER:")
    blend_gate_found = False
    for name, param in model.named_parameters():
        if 'blend' in name.lower():
            print(f"   ✅ Found: {name}, shape={param.shape}, value={param.item():.4f}")
            blend_gate_found = True
    
    if not blend_gate_found:
        print("   ⚠️  Blend gate NOT found (will be created lazily on first forward)")
    
    # Run forward pass to trigger lazy init
    print("\n2. RUNNING FORWARD PASS:")
    batch_size = 4
    x = torch.randn(batch_size, args.seq_len, args.enc_in)
    x_mark_enc = torch.randn(batch_size, args.seq_len, 4)
    x_dec = torch.randn(batch_size, args.pred_len, args.dec_in)
    x_mark_dec = torch.randn(batch_size, args.pred_len, 4)
    
    outputs = model(x, x_mark_enc, x_dec, x_mark_dec)
    print(f"   ✅ Forward pass completed, output shape: {outputs.shape}")
    
    # Check blend_gate again after forward
    print("\n3. CHECKING ENCODER_BLEND_GATE AFTER FORWARD:")
    for name, param in model.named_parameters():
        if 'blend' in name.lower():
            print(f"   ✅ Found: {name}, value={param.item():.4f}")
            blend_gate_found = True
    
    if not blend_gate_found:
        print("   ❌ ERROR: Blend gate still not found after forward pass!")
    
    # Compute loss and backward to get gradients
    print("\n4. RUNNING BACKWARD PASS:")
    loss = outputs.mean()
    loss.backward()
    print("   ✅ Backward pass completed")
    
    # Check spatiotemporal encoder gradients
    print("\n5. CHECKING SPATIOTEMPORAL ENCODER GRADIENTS:")
    spatiotemporal_grads_found = False
    zero_grads = []
    nonzero_grads = []
    
    for name, param in model.named_parameters():
        if 'spatiotemporal_encoder' in name and param.grad is not None:
            spatiotemporal_grads_found = True
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-8:
                nonzero_grads.append((name, grad_norm))
            else:
                zero_grads.append(name)
    
    if nonzero_grads:
        print(f"   ✅ Found {len(nonzero_grads)} parameters with NON-ZERO gradients:")
        for name, grad_norm in nonzero_grads[:5]:  # Show first 5
            print(f"      {name}: grad_norm={grad_norm:.8f}")
        if len(nonzero_grads) > 5:
            print(f"      ... and {len(nonzero_grads) - 5} more")
    
    if zero_grads:
        print(f"   ⚠️  Found {len(zero_grads)} parameters with ZERO gradients:")
        for name in zero_grads[:3]:
            print(f"      {name}")
        if len(zero_grads) > 3:
            print(f"      ... and {len(zero_grads) - 3} more")
    
    if not spatiotemporal_grads_found:
        print("   ❌ ERROR: No spatiotemporal_encoder parameters found with gradients!")
    
    # Check blend_gate gradient
    print("\n6. CHECKING ENCODER_BLEND_GATE GRADIENT:")
    for name, param in model.named_parameters():
        if 'blend' in name.lower() and param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"   ✅ {name}: grad_norm={grad_norm:.8f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY:")
    print("=" * 70)
    if blend_gate_found:
        print("✅ Encoder blend gate: CREATED")
    else:
        print("❌ Encoder blend gate: NOT FOUND")
    
    if nonzero_grads:
        print(f"✅ Spatiotemporal encoder gradients: NON-ZERO ({len(nonzero_grads)} params)")
    elif spatiotemporal_grads_found:
        print(f"❌ Spatiotemporal encoder gradients: ALL ZERO ({len(zero_grads)} params)")
    else:
        print("❌ Spatiotemporal encoder: NO GRADIENTS FOUND")
    
    if blend_gate_found and nonzero_grads:
        print("\n🎉 FIX VERIFIED: Soft blending is working correctly!")
    else:
        print("\n⚠️  FIX INCOMPLETE: Issues detected above")


if __name__ == "__main__":
    verify_encoder_fix()
