"""Test Migration Audit Script

Provides two complementary audit views to support the migration:

1) Mapping summary between legacy TestsModule and tests/modular (heuristic by stem)
2) Root-level legacy tests (not under tests/modular) classified for actions

Usage:
  python -m scripts.test_migration_audit [--markdown reports/test_migration_audit.md]

Outputs:
  - Prints a human-readable summary to stdout
  - Writes JSON to reports/test_migration_audit.json
  - Optionally writes a Markdown table when --markdown is provided
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = REPO_ROOT / "TestsModule"
MODULAR_DIR = REPO_ROOT / "tests" / "modular"
REPORTS_DIR = REPO_ROOT / "reports"
OUTPUT_JSON = REPORTS_DIR / "test_migration_audit.json"

# Heuristic classification for root-level tests
KEYWORD_MAP = {
    "probabilistic": {"quantile", "uncertainty", "bayesian"},
    "hierarchical": {"hierarchical", "wavelet", "multi_scale"},
    "hf_migration": {"huggingface", "hf_"},
    "chronosx": {"chronosx", "chronos_x"},
}
RUNNER_HINTS = {"test_sequence", "run_all_tests", "run_test_file"}


# ---------- Mapping between TestsModule and tests/modular ----------
def _collect_tests(base: Path) -> List[str]:
    return sorted(
        str(p.relative_to(base)).replace("\\", "/")
        for p in base.rglob("test_*.py")
        if p.is_file()
    )


def _heuristic_match(legacy_paths: List[str], modular_paths: List[str]) -> Dict[str, List[str]]:
    from collections import defaultdict

    modular_index: Dict[str, List[str]] = defaultdict(list)
    for m in modular_paths:
        stem = Path(m).stem
        modular_index[stem].append(m)

    mapping: Dict[str, List[str]] = {}
    for l in legacy_paths:
        stem = Path(l).stem
        candidates = modular_index.get(stem, [])
        parent = Path(l).parts[0] if len(Path(l).parts) > 0 else ""
        ordered = sorted(
            candidates,
            key=lambda x: (0 if parent and parent in x else 1, len(x)),
        )
        mapping[l] = ordered
    return mapping


def build_mapping_summary() -> Dict[str, object]:
    if not LEGACY_DIR.exists():
        return {
            "legacy_total": 0,
            "modular_total": len(_collect_tests(MODULAR_DIR)) if MODULAR_DIR.exists() else 0,
            "legacy_without_match": 0,
            "estimated_coverage_percent": 100.0,
            "examples": {},
        }

    legacy = _collect_tests(LEGACY_DIR)
    modular = _collect_tests(MODULAR_DIR) if MODULAR_DIR.exists() else []
    mapping = _heuristic_match(legacy, modular)
    orphans = [k for k, v in mapping.items() if not v]
    coverage = 100.0 * (1.0 - (len(orphans) / max(1, len(legacy))))
    return {
        "legacy_total": len(legacy),
        "modular_total": len(modular),
        "legacy_without_match": len(orphans),
        "estimated_coverage_percent": round(coverage, 1),
        "examples": {k: mapping[k][:3] for k in legacy[:10]},
    }


# ---------- Root-level legacy tests classification ----------
@dataclass
class TestFileInfo:
    file: str
    size: int
    prints: int
    keywords: Set[str]
    categories: List[str]
    suggested_action: str

    def to_row(self) -> str:
        return (
            f"| {self.file} | {','.join(self.categories)} | {self.size} | {self.prints} | "
            f"{','.join(sorted(self.keywords))} | {self.suggested_action} |"
        )

    def to_json(self) -> dict:
        return {
            "file": self.file,
            "size": self.size,
            "prints": self.prints,
            "keywords": sorted(self.keywords),
            "categories": self.categories,
            "suggested_action": self.suggested_action,
        }


def discover_legacy_tests() -> List[Path]:
    candidates: List[Path] = []
    for path in REPO_ROOT.glob("test_*.py"):
        if str(path).startswith(str(MODULAR_DIR)):
            continue
        candidates.append(path)
    return sorted(candidates)


def classify(file_path: Path) -> TestFileInfo:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    prints = sum(1 for l in lines if "print(" in l)
    lower_text = text.lower()

    matched_keywords: Set[str] = set()
    categories: Set[str] = set()
    for cat, kws in KEYWORD_MAP.items():
        if any(kw in lower_text for kw in kws):
            categories.add(cat)
            matched_keywords.update({kw for kw in kws if kw in lower_text})

    name = file_path.name
    if name.startswith("test_step") or name.startswith("test_phase"):
        categories.add("phased_upgrade")
    if "moe" in name:
        categories.add("algorithmic")
    if prints > 25 or any(h in lower_text for h in RUNNER_HINTS):
        categories.add("legacy_redundant")
    if not categories:
        categories.add("integration")

    if "phased_upgrade" in categories:
        suggested = "merge"
    elif "legacy_redundant" in categories:
        suggested = "prune"
    elif "probabilistic" in categories or "hierarchical" in categories:
        suggested = "merge"
    else:
        suggested = "review"

    return TestFileInfo(
        file=name,
        size=len(lines),
        prints=prints,
        keywords=matched_keywords,
        categories=sorted(categories),
        suggested_action=suggested,
    )


def emit_markdown(infos: List[TestFileInfo]) -> str:
    header = (
        "| File | Categories | Lines | Prints | Keywords | Suggested |\n"
        "|------|-----------|-------|--------|----------|-----------|"
    )
    rows = [info.to_row() for info in infos]
    return "\n".join([header, *rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit legacy test files and migration mapping")
    parser.add_argument(
        "--markdown",
        dest="markdown_path",
        help="Optional path to also write the markdown summary table",
        default=None,
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = parse_args()

    # Build mapping summary
    mapping_summary = build_mapping_summary()

    # Classify root-level legacy tests
    legacy_files = discover_legacy_tests()
    infos = [classify(p) for p in legacy_files]

    # Ensure reports dir and write JSON
    REPORTS_DIR.mkdir(exist_ok=True)
    payload = {
        "mapping_summary": mapping_summary,
        "root_tests_audit": [i.to_json() for i in infos],
    }
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Emit human-readable output
    print("# Test Migration Audit\n")
    print(f"Repository Root: {REPO_ROOT}")
    print("\n[Mapping Summary: TestsModule -> tests/modular]")
    print(json.dumps(mapping_summary, indent=2))

    print("\n[Root-level Legacy Tests Classification]\n")
    print(f"Legacy root-level tests detected: {len(infos)}\n")
    md = emit_markdown(infos)
    print(md)
    print("\n(JSON written to reports/test_migration_audit.json)")
    if args.markdown_path:
        md_path = Path(args.markdown_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        print(f"(Markdown table also written to {md_path})")


if __name__ == "__main__":
    main()
