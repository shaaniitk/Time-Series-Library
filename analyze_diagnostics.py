
import torch
import os
import glob
import numpy as np

def analyze_snapshot(run_dir):
    print(f"Analyzing run: {run_dir}")
    
    # Get all steps
    files = glob.glob(os.path.join(run_dir, "c2t_gate_target_0_step_*.pt"))
    steps = sorted([int(f.split('_step_')[-1].replace('.pt', '')) for f in files])
    
    if not steps:
        print("No diagnostic files found.")
        return

    print(f"Found steps: {steps}")
    
    # Analyze Gates
    print("\n--- Gate Analysis (Fusion Gates) ---")
    for target in range(4):
        print(f"\nTarget {target}:")
        
        start_step = steps[0]
        end_step = steps[-1]
        
        for step in [start_step, end_step]:
            fname = os.path.join(run_dir, f"c2t_gate_target_{target}_step_{step}.pt")
            if not os.path.exists(fname): continue
            
            gate = torch.load(fname)
            mean = gate.mean().item()
            std = gate.std().item()
            saturation_0 = (gate < 0.1).float().mean().item()
            saturation_1 = (gate > 0.9).float().mean().item()
            
            print(f"  Step {step}: Mean={mean:.4f}, Std={std:.4f}, Saturation(0)={saturation_0:.2%}, Saturation(1)={saturation_1:.2%}")

    # Analyze Attention
    print("\n--- Attention Analysis (Celestial Weights) ---")
    for target in range(4):
        print(f"\nTarget {target}:")
        
        for step in [start_step, end_step]:
            fname = os.path.join(run_dir, f"c2t_attn_target_{target}_step_{step}.pt")
            if not os.path.exists(fname): continue
            
            attn = torch.load(fname) # [Batch, PredLen, NumCelestial]
            
            # Mean attention over batch and time
            mean_attn = attn.mean(dim=(0,1)) # [NumCelestial]
            
            # Entropy
            eps = 1e-8
            entropy = -(attn * torch.log(attn + eps)).sum(dim=-1).mean().item()
            
            print(f"  Step {step}: Entropy={entropy:.4f}")
            print(f"    Top 3 Influential Bodies: {torch.topk(mean_attn, 3).indices.tolist()}")
            print(f"    Top 3 Weights: {torch.topk(mean_attn, 3).values.tolist()}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        # Find latest run automatically
        runs = sorted(glob.glob("diagnostics/run_*"))
        if not runs:
            print("No runs found in diagnostics/")
            exit(1)
        run_dir = runs[-1]
    
    analyze_snapshot(run_dir)
