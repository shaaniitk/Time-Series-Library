#!/usr/bin/env python3
"""
Run diagnostic training with modified config.
This will run 1 epoch and generate training_diagnostic.log
"""

# Modify the training script to use diagnostic config
import os

# Read the original training script
with open('scripts/train/train_celestial_production.py', 'r') as f:
    script_content = f.read()

# Replace the config path
modified_script = script_content.replace(
    'config_path = "configs/celestial_enhanced_pgat_production.yaml"',
    'config_path = "configs/celestial_diagnostic.yaml"'
)

# Execute the modified script
print("="*80)
print("STARTING DIAGNOSTIC TRAINING RUN (1 EPOCH)")
print("="*80)
print("\nThis will generate:")
print("  - training_diagnostic.log (detailed batch-level diagnostics)")
print("  - Standard training logs in logs/ directory")
print("\n" + "="*80 + "\n")

exec(modified_script)

print("\n" + "="*80)
print("DIAGNOSTIC TRAINING COMPLETE")
print("="*80)
print("\nTo view the diagnostic log:")
print("  cat training_diagnostic.log")
print("  # or")
print("  head -500 training_diagnostic.log")
