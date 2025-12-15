"""Move DELETE-classified trivial test files to a quarantine deletion queue.

Reads internal_test_inventory_stage3.json and relocates files whose
classification == 'DELETE' into TestsModule/quarantine/delete_later/ keeping
original relative path encoded in a manifest for traceability.

We intentionally move (not copy) to reduce noise in active test discovery. The
quarantine folder can be excluded from routine CI runs later or removed at the
final cleanup stage.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import List, Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INVENTORY_FILENAME = "internal_test_inventory_stage3.json"
QUARANTINE_DIR = os.path.join(REPO_ROOT, "TestsModule", "quarantine", "delete_later")
MANIFEST = os.path.join(QUARANTINE_DIR, "deletion_candidates_stage3.json")


def _load_inventory() -> List[Dict[str, Any]]:
    """Load only the JSON portion of the inventory file (strip guidance)."""
    inv_path = os.path.join(REPO_ROOT, INVENTORY_FILENAME)
    with open(inv_path, "r", encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()
    for sentinel in ["\nSPLIT_RECOMMENDATIONS", "\nDELETE_CANDIDATES", "\nSEMANTIC_DUPLICATE_GROUPS"]:
        idx = raw.find(sentinel)
        if idx != -1:
            trimmed = raw[:idx].rstrip()
            last_brace = trimmed.rfind('}')
            if last_brace != -1:
                raw = trimmed[: last_brace + 1]
            break
    data = json.loads(raw)
    return data.get("file_summaries", [])


def main() -> None:
    os.makedirs(QUARANTINE_DIR, exist_ok=True)
    summaries = _load_inventory()
    moved = []
    for s in summaries:
        if s.get("classification") != "DELETE":
            continue
        rel = s["rel_path"]
        src = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(src):  # Already moved or missing
            continue
        base_name = os.path.basename(rel)
        dest = os.path.join(QUARANTINE_DIR, base_name)
        if os.path.exists(dest):  # collision safeguard
            stem, ext = os.path.splitext(base_name)
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(QUARANTINE_DIR, f"{stem}_{counter}{ext}")
                counter += 1
        try:
            shutil.move(src, dest)
        except Exception as exc:  # pragma: no cover
            print(f"WARN: failed to move {src} -> {dest}: {exc}")
            continue
        moved.append({
            "original_rel_path": rel.replace("\\", "/"),
            "new_rel_path": os.path.relpath(dest, REPO_ROOT).replace("\\", "/"),
            "lines": s.get("lines"),
            "asserts": s.get("asserts"),
            "test_functions": s.get("test_functions"),
            "reason": "DELETE classification stage3 (trivial or empty)"
        })
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump({"moved_count": len(moved), "entries": moved}, fh, indent=2)
    print(f"Moved {len(moved)} deletion candidates -> {QUARANTINE_DIR}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":  # pragma: no cover
    main()
