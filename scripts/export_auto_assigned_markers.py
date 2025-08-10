"""Utility to export auto-assigned test markers to JSON for CI artifacts.

Run after a pytest collection phase that may auto-assign markers.
We re-invoke pytest in --collect-only mode to recompute enforcement logic
and then write any auto-assigned decisions into .test_baseline/auto_assigned_markers.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

def main() -> int:
    # We rely on pytest writing terminal lines; to capture auto-assigned we pass env to force classification.
    # Simpler: run pytest -q to trigger logic, then parse the stats file we stash in config via a plugin hook.
    # For now we make a specialized run that sets AUTO_CLASSIFY_MISSING=1.
    env = dict(**{k: v for k, v in dict(**os.environ).items()})  # copy
    env.setdefault("AUTO_CLASSIFY_MISSING", "1")
    # Use a very narrow marker selection to keep it quick.
    cmd = [sys.executable, "-m", "pytest", "-m", "smoke", "--maxfail=1", "-q"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # Extract auto-assigned summary line
    auto_line = None
    for line in proc.stdout.splitlines():
        if line.startswith("[marker-enforcement] Auto-assigned markers"):
            auto_line = line
            break
    baseline_dir = Path('.test_baseline')
    baseline_dir.mkdir(exist_ok=True)
    out_file = baseline_dir / 'auto_assigned_markers.json'
    payload = {"auto_assigned_summary": auto_line, "returncode": proc.returncode}
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_file} (returncode={proc.returncode})")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
