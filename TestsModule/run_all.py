"""Runner: execute all tests under TestsModule.
Usage: python TestsModule/run_all.py [additional pytest args]
"""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

THIS_DIR = Path(__file__).parent

def main(argv: list[str]) -> int:
    cmd = [sys.executable, '-m', 'pytest', str(THIS_DIR)] + argv
    return subprocess.call(cmd)

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
