#!/usr/bin/env python3
"""Test model.to(device) operation"""

import sys
sys.path.insert(0, '.')

import yaml
import torch
from scripts.train.train_celestial_production import SimpleConfig
from models.Celestial_Enhanced_PGAT_Modular import Model

print("Loading config...")
with open('configs/celestial_production_deep_ultimate_fixed.yaml') as f:
    config_dict = yaml.safe_load(f)

args = SimpleConfig(config_dict)
args.task_name = 'long_term_forecast'
args.model_name = 'Celestial_Enhanced_PGAT'
args.data_name = 'custom'
args.checkpoints = './checkpoints/'
args.inverse = False
args.cols = None

print("Creating model...")
model = Model(args)
print("✓ Model created successfully")

# Determine device
if getattr(args, 'use_gpu', True) and torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using device: cuda")
else:
    device = torch.device('cpu')
    print(f"Using device: cpu")

print(f"Moving model to {device}...")
try:
    model = model.to(device)
    print(f"✓✓✓ Model successfully moved to {device} ✓✓✓")
except Exception as e:
    print(f"ERROR during model.to(device): {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("SUCCESS - Model is ready for training!")
