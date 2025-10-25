#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke Test Runner for Phase 1 MDN Decoder Integration

Runs minimal 2-epoch training to validate:
- MDN decoder pipeline
- Loss computation (MDN NLL)
- Calibration metrics logging
- End-to-end training stability

Usage:
    python test_scripts/smoke_test_mdn_training.py
"""
import sys
import os
import time

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

sys.path.insert(0, os.path.abspath('.'))

print("="*80)
print("PHASE 1 MDN DECODER - SMOKE TEST")
print("="*80)
print("\nTest Configuration:")
print("   - Epochs: 2 (quick validation)")
print("   - Batch size: 4")
print("   - Sequence: 30 -> 5 prediction")
print("   - Model: d_model=64, n_heads=4, e_layers=2")
print("   - MDN: enabled, K=3 components")
print("   - Target: Verify pipeline stability, NOT convergence")
print("="*80)

# Run training
start_time = time.time()

os.system("""
python scripts/train/train_celestial_production.py \
    --config configs/test_celestial_smoke.yaml \
    --model Celestial_Enhanced_PGAT \
    --data custom
""")

elapsed = time.time() - start_time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)

print("\n" + "="*80)
print(f"Smoke test completed in {minutes}m {seconds}s")
print("="*80)
print("\nNext steps:")
print("   1. Check logs for 'MDN CALIBRATION' metrics")
print("   2. Verify training loss is finite (NLL)")
print("   3. Check checkpoint saved successfully")
print("\nCommands:")
print("   grep 'MDN CALIBRATION' logs/memory_diagnostics_*.log")
print("   grep 'Train Loss' logs/*.log | tail -5")
print("="*80)
