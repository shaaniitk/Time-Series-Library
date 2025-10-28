#!/usr/bin/env python3
"""
Debug script to analyze why training loss plateaus after epoch 6
"""

import torch
import torch.nn as nn
from pathlib import Path
import json

print("=" * 80)
print("TRAINING PLATEAU DIAGNOSTIC")
print("=" * 80)

# Check if there are any recent checkpoints
checkpoint_dirs = list(Path("checkpoints").glob("celestial_enhanced_pgat_production*"))
if checkpoint_dirs:
    latest_checkpoint_dir = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
    print(f"\nLatest checkpoint directory: {latest_checkpoint_dir}")
    
    # Check for results file
    results_file = latest_checkpoint_dir / "production_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print("\nTraining Loss History:")
        train_losses = results.get('train_losses', [])
        val_losses = results.get('val_losses', [])
        
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            delta = ""
            if i > 0:
                train_delta = train_loss - train_losses[i-1]
                delta = f" (Δ={train_delta:+0.6f})"
            print(f"  Epoch {i+1:2d}: train={train_loss:.6f}{delta:20s} val={val_loss:.6f}")
        
        # Analyze the plateau
        if len(train_losses) > 6:
            print("\nPlateau Analysis:")
            epoch_6_loss = train_losses[5]  # 0-indexed, so epoch 6 is index 5
            deltas_after_6 = [train_losses[i] - train_losses[i-1] for i in range(6, len(train_losses))]
            
            avg_delta = sum(abs(d) for d in deltas_after_6) / len(deltas_after_6) if deltas_after_6 else 0
            print(f"  Average absolute change after epoch 6: {avg_delta:.8f}")
            
            if avg_delta < 1e-4:
                print("  ⚠️  CONFIRMED: Loss is essentially flat after epoch 6")
                print("\nPossible causes:")
                print("  1. Learning rate too small during warmup")
                print("  2. Gradient clipping too aggressive (clip_grad_norm=1.0)")
                print("  3. Weight decay too high (0.0001)")
                print("  4. Vanishing gradients in deep model (8 encoder layers)")
                print("  5. Optimizer state issue (Adam adaptive LR stuck)")
            else:
                print(f"  ✓ Loss is still changing (avg |Δ|={avg_delta:.8f})")
    
    # Check for checkpoint files
    checkpoints = list(latest_checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if checkpoints:
        print(f"\nFound {len(checkpoints)} checkpoint files")
        
        # Load a couple of checkpoints to check optimizer state
        checkpoint_6 = latest_checkpoint_dir / "checkpoint_epoch_6.pth"
        checkpoint_7 = latest_checkpoint_dir / "checkpoint_epoch_7.pth"
        
        if checkpoint_6.exists() and checkpoint_7.exists():
            print("\nComparing optimizer states between epochs 6 and 7:")
            
            cp6 = torch.load(checkpoint_6, map_location='cpu')
            cp7 = torch.load(checkpoint_7, map_location='cpu')
            
            print(f"  Epoch 6 LR: {cp6.get('lr', 'N/A')}")
            print(f"  Epoch 7 LR: {cp7.get('lr', 'N/A')}")
            print(f"  Epoch 6 train_loss: {cp6.get('train_loss', 'N/A')}")
            print(f"  Epoch 7 train_loss: {cp7.get('train_loss', 'N/A')}")
            print(f"  Epoch 6 val_loss: {cp6.get('val_loss', 'N/A')}")
            print(f"  Epoch 7 val_loss: {cp7.get('val_loss', 'N/A')}")
            
            # Check if optimizer state has reasonable values
            opt6 = cp6.get('optimizer_state_dict', {})
            opt7 = cp7.get('optimizer_state_dict', {})
            
            if 'state' in opt6 and 'state' in opt7:
                # Check first parameter's Adam state
                if opt6['state'] and opt7['state']:
                    first_key = list(opt6['state'].keys())[0]
                    
                    exp_avg_6 = opt6['state'][first_key].get('exp_avg', None)
                    exp_avg_7 = opt7['state'][first_key].get('exp_avg', None)
                    
                    if exp_avg_6 is not None and exp_avg_7 is not None:
                        change = (exp_avg_7 - exp_avg_6).abs().mean().item()
                        print(f"\n  Adam momentum change (exp_avg): {change:.8f}")
                        if change < 1e-6:
                            print("  ⚠️  WARNING: Adam momentum barely changing - optimizer might be stuck")
else:
    print("\nNo checkpoint directories found in ./checkpoints/")
    print("Please run a training session first.")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print("""
1. Check the learning rate schedule:
   - warmup_epochs=8 means LR ramps from 0.000125 to 0.001 over 8 epochs
   - At epoch 6, LR should be ~0.000875 (7/8 of base_lr)
   - This might be too small for a deep model to learn effectively

2. Try these config changes:
   - Reduce warmup_epochs from 8 to 3 or 5
   - Increase learning_rate from 0.001 to 0.002 or 0.003
   - Reduce clip_grad_norm from 1.0 to 5.0 or 10.0
   - Reduce weight_decay from 0.0001 to 0.00005 or 0.00001

3. Check if mixed precision is helping:
   - Currently mixed_precision=true
   - Try with false to see if numerical issues are causing problems

4. Verify gradients are flowing:
   - Check training_diagnostic.log for grad_norm values
   - If grad_norms are very small (<1e-5), gradients are vanishing
   - If grad_norms are very large (>100), gradients are exploding
""")

print("Run this script after your next training session to see detailed diagnostics.")
