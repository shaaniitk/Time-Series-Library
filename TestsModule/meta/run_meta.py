"""Run meta tests (analyzer health, registry metadata) when added."""
from __future__ import annotations
import sys, subprocess, pathlib

BASE = pathlib.Path(__file__).parent

def main(argv: list[str]) -> int:
    return subprocess.call([sys.executable, '-m', 'pytest', str(BASE)] + argv)

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
