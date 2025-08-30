from __future__ import annotations
from typing import List

def build_row(nodeid: str, duration: float, status: str, markers: List[str]) -> List[str]:
    """Lightweight fallback used by root conftest when TestsModule helpers aren't available."""
    return [nodeid, f"{duration:.3f}", status, ",".join(markers)]


def append_rows(rows: List[List[str]]) -> None:
    """No-op append for environments without a persistent results sink."""
    # Intentionally do nothing; main CI may override this via TestsModule helpers.
    return
