"""Runner for just smoke tests under TestsModule."""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

THIS_DIR = Path(__file__).parent.parent

def main(argv: list[str]) -> int:
    cmd = [sys.executable, '-m', 'pytest', '-m', 'smoke', str(THIS_DIR)] + argv
    return subprocess.call(cmd)

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
