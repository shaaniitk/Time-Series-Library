#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Smoke Test with Full Logging - Phase 1-3 Validation

Runs minimal 2-epoch training with FULL LOGGING to validate:
- Edge feature stabilization (velocity_ratio, radius_ratio)
- Phase 2/3 bug fixes (Adaptive TopK, Stochastic Control)
- MDN decoder pipeline
- End-to-end training stability

Usage:
    python run_smoke_with_logs.py
"""
import sys
import os
import time
import subprocess
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

# Ensure we're in the correct directory
os.chdir(Path(__file__).parent)

print("=" * 80)
print("🚀 ENHANCED SMOKE TEST - FULL LOGGING ENABLED")
print("=" * 80)
print("\n📋 Test Configuration:")
print("   - Epochs: 2 (quick validation)")
print("   - Batch size: 4")
print("   - Sequence: 30 -> 5 prediction")
print("   - Model: d_model=64, n_heads=4, e_layers=2")
print("   - MDN: enabled, K=3 components")
print("   - Logging: VERBOSE (all diagnostics enabled)")
print("\n🎯 Validation Targets:")
print("   ✅ Edge feature stabilization (tanh on velocity/radius ratios)")
print("   ✅ Phase 2/3 bug fixes (11 critical bugs)")
print("   ✅ MDN decoder pipeline stability")
print("   ✅ Gradient flow through attention")
print("=" * 80)

# Create logs directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Generate timestamp for log files
timestamp = time.strftime("%Y%m%d_%H%M%S")
stdout_log = log_dir / f"smoke_test_stdout_{timestamp}.log"
stderr_log = log_dir / f"smoke_test_stderr_{timestamp}.log"

print(f"\n📝 Log files:")
print(f"   STDOUT: {stdout_log}")
print(f"   STDERR: {stderr_log}")
print(f"   Memory: logs/memory_diagnostics_*.log")
print("\n⏳ Starting training...\n")

# Run training with full logging
start_time = time.time()

# Build command
cmd = [
    sys.executable,
    "scripts/train/train_celestial_production.py",
    "--config", "configs/test_celestial_smoke_verbose.yaml",
    "--model", "Celestial_Enhanced_PGAT",
    "--data", "custom"
]

# Run with live output to terminal AND log files
with open(stdout_log, 'w', encoding='utf-8') as out_f, \
     open(stderr_log, 'w', encoding='utf-8') as err_f:
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        bufsize=1
    )
    
    # Print output in real-time while also writing to log
    import threading
    
    def stream_output(stream, file_obj, prefix=""):
        for line in stream:
            print(f"{prefix}{line}", end='')
            file_obj.write(line)
            file_obj.flush()
    
    stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, out_f, ""))
    stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, err_f, "⚠️  "))
    
    stdout_thread.start()
    stderr_thread.start()
    
    stdout_thread.join()
    stderr_thread.join()
    
    return_code = process.wait()

elapsed = time.time() - start_time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)

print("\n" + "=" * 80)
if return_code == 0:
    print(f"✅ Smoke test COMPLETED in {minutes}m {seconds}s")
else:
    print(f"❌ Smoke test FAILED with return code {return_code} ({minutes}m {seconds}s)")
print("=" * 80)

print("\n🔍 Next steps - Check logs for:")
print("   1. Edge features: grep 'stabilization_applied' logs/smoke_test_stdout_*.log")
print("   2. MDN calibration: grep 'MDN CALIBRATION' logs/*.log")
print("   3. Training loss: grep 'Train Loss' logs/smoke_test_stdout_*.log | tail -5")
print("   4. Phase metadata: grep 'avg_velocity_ratio\\|avg_radius_ratio' logs/*.log")
print("\n📊 Quick analysis:")
print(f"   tail -n 50 {stdout_log}")
print(f"   grep -i 'error\\|warning\\|fail' {stderr_log}")
print("=" * 80)

sys.exit(return_code)
