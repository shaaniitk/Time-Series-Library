#!/usr/bin/env python3
"""
Minimal diagnostic script to check train vs val loss.
Runs exactly 1 epoch with detailed logging.
"""

import sys
import subprocess

print("="*80)
print("DIAGNOSTIC TRAINING - 1 EPOCH WITH DETAILED LOGGING")
print("="*80)

# First, modify the config to use just 1 epoch
import yaml

config_file = "configs/celestial_enhanced_pgat_production.yaml"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Save original epochs
original_epochs = config['train_epochs']
print(f"\nOriginal train_epochs: {original_epochs}")

# Modify to 1 epoch for diagnostic
config['train_epochs'] = 1
config['patience'] = 10
config['log_interval'] = 1

# Save as diagnostic config
diagnostic_config = "configs/celestial_diagnostic.yaml"
with open(diagnostic_config, 'w') as f:
    yaml.safe_dump(config, f)

print(f"Created diagnostic config: {diagnostic_config}")
print(f"Running 1 epoch with detailed logging...")
print("="*80)
print()

# Now run the training script
# We need to modify it to use our diagnostic config
# Let's just call it directly with Python
try:
    exec(open("scripts/train/train_celestial_production.py").read().replace(
        'config_path = "configs/celestial_enhanced_pgat_production.yaml"',
        'config_path = "configs/celestial_diagnostic.yaml"'
    ))
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
print("\nCheck the following files:")
print("  - training_diagnostic.log (detailed batch-level diagnostics)")
print("  - logs/ directory for full training logs")
