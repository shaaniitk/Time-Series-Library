"""Generate structured split plan for large monolithic test files.

Reads the stage3 inventory JSON (internal_test_inventory_stage3.json) and
produces a machine & human friendly plan file (split_plan_stage3.json) that
lists each SPLIT-classified source test with rationale and proposed new
modular target test file stubs under the TestsModule hierarchy.

Heuristics:
  * Use domain_keywords (if present) to derive thematic split buckets.
  * Fallback to generic buckets when domain coverage is sparse.
  * For very large files (>650 lines) recommend 4+ splits, else 2-3.
  * Provide a rationale string capturing size, asserts, heavy indicators.

The output JSON schema (per entry):
  {
    "source": str,
    "lines": int,
    "asserts": int,
    "heavy_indicators": int,
    "domain_keywords": [str],
    "recommended_splits": [
       {"rel_path": str, "focus": str, "marker": str}
    ],
    "rationale": str
  }

This script is idempotent and safe to re-run; it overwrites the plan file.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import List, Dict, Any

INVENTORY_FILENAME = "internal_test_inventory_stage3.json"
OUTPUT_FILENAME = "split_plan_stage3.json"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class FileSummary:
    rel_path: str
    lines: int
    asserts: int
    test_functions: int
    import_domains: int
    heavy_indicators: int
    category: str
    classification: str
    domain_keywords: List[str]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FileSummary":
        return cls(
            rel_path=d["rel_path"],
            lines=d.get("lines", 0),
            asserts=d.get("asserts", 0),
            test_functions=d.get("test_functions", 0),
            import_domains=d.get("import_domains", 0),
            heavy_indicators=d.get("heavy_indicators", 0),
            category=d.get("category", ""),
            classification=d.get("classification", "KEEP"),
            domain_keywords=d.get("domain_keywords", []) or [],
        )


def _load_inventory() -> List[FileSummary]:
    """Load the inventory JSON, trimming any appended human guidance sections.

    The analyzer prints JSON followed by plain-text guidance (SPLIT_RECOMMENDATIONS etc.).
    We cut off at the first occurrence of a guidance sentinel and then parse.
    """
    path = os.path.join(REPO_ROOT, INVENTORY_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Inventory file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()
    # Sentinels that indicate non-JSON trailing content
    for sentinel in ["\nSPLIT_RECOMMENDATIONS", "\nDELETE_CANDIDATES", "\nSEMANTIC_DUPLICATE_GROUPS"]:
        idx = raw.find(sentinel)
        if idx != -1:
            # Attempt to clip at the end of the JSON object preceding sentinel
            trimmed = raw[:idx].rstrip()
            # Ensure ends with closing brace
            last_brace = trimmed.rfind('}')
            if last_brace != -1:
                raw = trimmed[: last_brace + 1]
            break
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to parse inventory JSON (trimmed) at {path}: {e}")
    summaries = [FileSummary.from_dict(d) for d in data.get("file_summaries", [])]
    return summaries


def _focus_labels(keywords: List[str], desired: int) -> List[str]:
    """Derive human friendly focus labels from domain keywords.

    We cluster by simple ordering; if insufficient keywords we extend with generic buckets.
    """
    base = []
    seen = set()
    for kw in keywords:
        if kw in seen:
            continue
        seen.add(kw)
        base.append(kw)
        if len(base) >= desired:
            break
    GENERIC = ["basics", "configurations", "edge_cases", "integration", "performance"]
    for g in GENERIC:
        if len(base) >= desired:
            break
        if g not in seen:
            base.append(g)
    return base[:desired]


def build_plan(summaries: List[FileSummary]) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for s in summaries:
        if s.classification != "SPLIT":
            continue
        # Determine number of splits
        if s.lines >= 700 or (s.lines >= 500 and s.heavy_indicators > 0):
            target_parts = 4
        elif s.lines >= 550:
            target_parts = 3
        else:
            target_parts = 2
        focus_labels = _focus_labels(s.domain_keywords, target_parts)
        recommended: List[Dict[str, Any]] = []
        base_name = os.path.splitext(os.path.basename(s.rel_path))[0]
        for idx, focus in enumerate(focus_labels, start=1):
            new_name = f"{base_name}_{focus}.py".replace("-", "_")
            # Choose destination folder heuristically
            if s.category in {"integration", "model", "chronosx"} or s.heavy_indicators > 0:
                dest_dir = os.path.join("TestsModule", "integration")
            else:
                dest_dir = os.path.join("TestsModule", "components", s.category or "misc")
            rel_new = os.path.join(dest_dir, new_name)
            marker = "extended" if s.lines > 400 else "smoke"
            recommended.append({"rel_path": rel_new.replace("\\", "/"), "focus": focus, "marker": marker})
        rationale = (
            f"File {s.rel_path} has {s.lines} lines, {s.asserts} asserts, "
            f"heavy_indicators={s.heavy_indicators}. Split into {len(recommended)} focused files to improve maintainability."
        )
        plan.append({
            "source": s.rel_path.replace("\\", "/"),
            "lines": s.lines,
            "asserts": s.asserts,
            "heavy_indicators": s.heavy_indicators,
            "domain_keywords": s.domain_keywords,
            "recommended_splits": recommended,
            "rationale": rationale,
        })
    return plan


def save_plan(plan: List[Dict[str, Any]]) -> str:
    out_path = os.path.join(REPO_ROOT, OUTPUT_FILENAME)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"plan_version": 1, "entries": plan}, fh, indent=2)
    return out_path


def main() -> None:
    summaries = _load_inventory()
    plan = build_plan(summaries)
    path = save_plan(plan)
    print(f"Wrote split plan with {len(plan)} entries -> {path}")


if __name__ == "__main__":  # pragma: no cover - CLI only
    main()
