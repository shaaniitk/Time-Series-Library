"""
Simple verification: Load the trained checkpoint and check for blend_gate and gradients.
"""

import torch


def verify_checkpoint() -> None:
    """Verify the checkpoint has blend_gate and spatiotemporal params."""
    checkpoint_path = "checkpoints/celestial_enhanced_pgat_production_overnight/best_model.pth"
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    print(f"\nTotal parameters in checkpoint: {len(state_dict)}\n")
    
    # Check for blend_gate
    print("1. CHECKING FOR ENCODER_BLEND_GATE:")
    blend_found = False
    for key in state_dict.keys():
        if 'blend' in key.lower():
            print(f"   ✅ Found: {key}")
            print(f"      Value: {state_dict[key].item():.4f}")
            blend_found = True
    
    if not blend_found:
        print("   ❌ No blend_gate parameter found in checkpoint")
    
    # Check for spatiotemporal encoder params
    print("\n2. CHECKING FOR SPATIOTEMPORAL_ENCODER PARAMETERS:")
    spatiotemporal_params = []
    for key in state_dict.keys():
        if 'spatiotemporal_encoder' in key:
            spatiotemporal_params.append(key)
    
    if spatiotemporal_params:
        print(f"   ✅ Found {len(spatiotemporal_params)} spatiotemporal encoder parameters:")
        for param in spatiotemporal_params[:5]:
            print(f"      {param}")
        if len(spatiotemporal_params) > 5:
            print(f"      ... and {len(spatiotemporal_params) - 5} more")
    else:
        print("   ❌ No spatiotemporal encoder parameters found")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    if blend_found:
        print("✅ Encoder blend gate: PRESENT in checkpoint")
    else:
        print("⚠️  Encoder blend gate: NOT FOUND (may indicate lazy init didn't trigger)")
    
    if spatiotemporal_params:
        print(f"✅ Spatiotemporal encoder: {len(spatiotemporal_params)} parameters present")
    else:
        print("❌ Spatiotemporal encoder: No parameters found")


if __name__ == "__main__":
    verify_checkpoint()
