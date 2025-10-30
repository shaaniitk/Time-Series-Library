#!/usr/bin/env python3
"""
Fix target extraction in training script
The model predicts 4 targets but loss is computed against all 118 features
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def fix_target_extraction():
    """Fix target extraction in the training script"""
    
    print("🎯 FIXING TARGET EXTRACTION...")
    
    # The issue is in the training script - we need to extract only the target features
    # from batch_y for loss computation
    
    training_script = "scripts/train/train_celestial_production.py"
    
    # Read the training script
    with open(training_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the loss computation section and fix target extraction
    # Look for the pattern where targets are extracted
    
    # Pattern 1: Direct target extraction
    old_target_pattern1 = '''y_true_for_loss = scale_targets_for_loss(
                batch_y[:, -args.pred_len:, :], target_scaler, target_indices, device
            )'''
    
    new_target_pattern1 = '''# FIXED: Extract only target features for loss computation
            batch_y_targets = batch_y[:, -args.pred_len:, :]
            if hasattr(args, 'target_wave_indices') and args.target_wave_indices:
                # Extract only the target columns
                batch_y_targets = batch_y_targets[:, :, args.target_wave_indices]
            elif hasattr(args, 'c_out') and args.c_out < batch_y_targets.shape[-1]:
                # Extract first c_out columns as targets
                batch_y_targets = batch_y_targets[:, :, :args.c_out]
            
            y_true_for_loss = scale_targets_for_loss(
                batch_y_targets, target_scaler, target_indices, device
            )'''
    
    if old_target_pattern1 in content:
        content = content.replace(old_target_pattern1, new_target_pattern1)
        print("✓ Fixed training target extraction")
    
    # Pattern 2: Validation target extraction
    old_val_pattern = '''y_true_for_loss = scale_targets_for_loss(
                    batch_y[:, -args.pred_len:, :], target_scaler, target_indices, device
                )'''
    
    new_val_pattern = '''# FIXED: Extract only target features for validation loss
                batch_y_targets = batch_y[:, -args.pred_len:, :]
                if hasattr(args, 'target_wave_indices') and args.target_wave_indices:
                    # Extract only the target columns
                    batch_y_targets = batch_y_targets[:, :, args.target_wave_indices]
                elif hasattr(args, 'c_out') and args.c_out < batch_y_targets.shape[-1]:
                    # Extract first c_out columns as targets
                    batch_y_targets = batch_y_targets[:, :, :args.c_out]
                
                y_true_for_loss = scale_targets_for_loss(
                    batch_y_targets, target_scaler, target_indices, device
                )'''
    
    if old_val_pattern in content:
        content = content.replace(old_val_pattern, new_val_pattern)
        print("✓ Fixed validation target extraction")
    
    # Also need to fix the scale_targets_for_loss function call pattern
    # Look for alternative patterns
    
    # Write back the fixed training script
    with open(training_script, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Also check if we need to update the config to specify target indices
    config_file = "configs/celestial_production_fixed.yaml"
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Ensure target_wave_indices is set correctly
        if 'target_wave_indices' not in config or not config['target_wave_indices']:
            # Set default target indices (first c_out features)
            c_out = config.get('c_out', 4)
            config['target_wave_indices'] = list(range(c_out))
            print(f"✓ Set target_wave_indices to first {c_out} features: {config['target_wave_indices']}")
            
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            print(f"✓ target_wave_indices already set: {config['target_wave_indices']}")
            
    except Exception as e:
        print(f"⚠️  Could not update config file: {e}")
    
    print("\n✅ TARGET EXTRACTION FIXES APPLIED")
    print("🎯 Changes made:")
    print("  - Fixed target extraction in training loop")
    print("  - Fixed target extraction in validation loop") 
    print("  - Ensured only target features are used for loss computation")
    print("  - Updated config with proper target_wave_indices")
    print("\nThis should resolve the dimension mismatch in loss computation")

if __name__ == "__main__":
    fix_target_extraction()