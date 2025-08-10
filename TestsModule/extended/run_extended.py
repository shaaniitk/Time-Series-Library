"""Run extended tests (marker-based)."""
from __future__ import annotations
import sys, subprocess, pathlib

ROOT = pathlib.Path(__file__).parent.parent

def main(argv: list[str]) -> int:
    return subprocess.call([sys.executable, '-m', 'pytest', '-m', 'extended', str(ROOT)] + argv)

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
