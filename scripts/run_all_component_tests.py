from __future__ import annotations

"""Single entry point to run all component tests in the new testing framework.

Usage (Windows PowerShell):
  .\\tsl-env\\Scripts\\python.exe run_all_component_tests.py
"""

import sys
import subprocess
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).parent.resolve()
    tests_dir = repo_root / "TestsModule" / "components"
    if not tests_dir.exists():
        print(f"Tests directory not found: {tests_dir}")
        return 2

    # Prefer running with the current interpreter (venv recommended)
    cmd = [sys.executable, "-X", "utf8", "-m", "pytest", "-q", str(tests_dir), "-q"]
    print("Running:", " ".join(cmd))
    try:
        return subprocess.call(cmd, cwd=str(repo_root))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
