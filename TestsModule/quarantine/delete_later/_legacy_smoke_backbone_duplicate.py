from __future__ import annotations

"""Legacy duplicate (inactive) of active smoke backbone test.

Renamed to remove pytest collection (filename no longer matches test_*.py)
and eliminate import mismatches. File kept temporarily as a sentinel
and will be deleted after migration cleanup.
"""

__deprecated_duplicate__ = True  # sentinel flag

def _noop() -> None:  # pragma: no cover
    return None
