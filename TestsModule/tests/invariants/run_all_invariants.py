"""Sequential runner for all invariant test modules.

Searches this directory for test_*.py files (excluding itself) and invokes
pytest on each one individually so per-file isolation timings / diagnostics
are visible.

Usage:
  python -m tests.invariants.run_all_invariants

Pass extra pytest args after -- (e.g. -q):
  python -m tests.invariants.run_all_invariants -- -q
"""
from __future__ import annotations

import sys
import pathlib
import subprocess
from typing import List


def main() -> None:  # pragma: no cover - CLI utility
    root = pathlib.Path(__file__).parent
    test_files: List[pathlib.Path] = [
        p for p in root.glob("test_*.py") if p.name != "run_all_invariants.py"
    ]
    test_files.sort()
    extra = []
    if "--" in sys.argv:
        dash_index = sys.argv.index("--")
        extra = sys.argv[dash_index + 1 :]
    print(f"[INVARIANT_RUNNER] Discovered {len(test_files)} files")
    failures = 0
    for tf in test_files:
        cmd = [sys.executable, "-m", "pytest", str(tf)] + extra
        print(f"\n[INVARIANT_RUNNER] Running {tf.name}: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures += 1
    if failures:
        print(f"[INVARIANT_RUNNER] Completed with {failures} failing files")
        sys.exit(1)
    print("[INVARIANT_RUNNER] All invariant test files passed")


if __name__ == "__main__":  # pragma: no cover
    main()
