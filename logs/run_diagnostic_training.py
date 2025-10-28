#!/usr/bin/env python3
"""
Quick diagnostic training run - just 2 epochs to generate diagnostic logs.
This will help us understand the train vs val loss discrepancy.
"""

import sys
import os
from pathlib import Path

# Ensure we're in the right directory
os.chdir(Path(__file__).parent)

# Import the training function
from scripts.train.train_celestial_production import train_celestial_pgat_production

# Modify the config to run just 2 epochs for quick diagnostics
import yaml

config_path = "configs/celestial_enhanced_pgat_production.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Override to run just 2 epochs for diagnostic
config['train_epochs'] = 2
config['patience'] = 10  # Don't stop early

# Save modified config temporarily
temp_config_path = "configs/diagnostic_config.yaml"
with open(temp_config_path, 'w') as f:
    yaml.safe_dump(config, f)

print("="*80)
print("RUNNING DIAGNOSTIC TRAINING - 2 EPOCHS ONLY")
print("="*80)
print(f"Original config: {config_path}")
print(f"Diagnostic config: {temp_config_path}")
print(f"Diagnostic log will be written to: training_diagnostic.log")
print("="*80)

# Temporarily modify the train function to use our config
import scripts.train.train_celestial_production as train_module
original_config_path = "configs/celestial_enhanced_pgat_production.yaml"
train_module.train_celestial_pgat_production.__globals__['config_path'] = temp_config_path

try:
    # Run training
    success = train_celestial_pgat_production()
    
    if success:
        print("\n" + "="*80)
        print("DIAGNOSTIC TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nDiagnostic log written to: training_diagnostic.log")
        print("\nTo view the diagnostic log:")
        print("  cat training_diagnostic.log | less")
        print("  # or")
        print("  head -200 training_diagnostic.log")
    else:
        print("\n" + "="*80)
        print("DIAGNOSTIC TRAINING FAILED")
        print("="*80)
        
except Exception as e:
    print(f"\nERROR during diagnostic training: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Cleanup temporary config
    if Path(temp_config_path).exists():
        Path(temp_config_path).unlink()
        print(f"\nCleaned up temporary config: {temp_config_path}")
