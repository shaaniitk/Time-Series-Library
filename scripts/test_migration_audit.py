"""Test Migration Audit Script

Scans legacy root-level test_*.py files (those NOT under tests/modular/) and emits:
1. A Markdown summary table (stdout)
2. A JSON artifact written to reports/test_migration_audit.json (created if missing)

Heuristic Classification (non-invasive):
 - Category inference based on keyword density:
     * probabilistic: 'quantile' or 'uncertainty' or 'bayesian'
     * hierarchical: 'hierarchical' or 'wavelet' or 'multi_scale'
     * phased_upgrade: filename starts with 'test_step' or 'test_phase'
     * hf_migration: 'hf_' or 'huggingface' in file content
     * chronosx: 'chronosx' or 'chronos_x'
     * legacy_redundant: heavy print usage (>25 print statements) or custom runner constructs
 - Multiple matches -> semicolon-joined categories

Fields Collected:
 - file
 - size (lines)
 - prints
 - keywords (matched set)
 - categories
 - suggested_action (migrate|merge|prune|review)

Does NOT modify repository; pure analysis aid for migration batching.
"""
from __future__ import annotations

import json
import re
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULAR_DIR = REPO_ROOT / "tests" / "modular"
REPORTS_DIR = REPO_ROOT / "reports"
OUTPUT_JSON = REPORTS_DIR / "test_migration_audit.json"

KEYWORD_MAP = {
    "probabilistic": {"quantile", "uncertainty", "bayesian"},
    "hierarchical": {"hierarchical", "wavelet", "multi_scale"},
    "hf_migration": {"huggingface", "hf_"},
    "chronosx": {"chronosx", "chronos_x"},
}

RUNNER_HINTS = {"test_sequence", "run_all_tests", "run_test_file"}


@dataclass
class TestFileInfo:
    file: str
    size: int
    prints: int
    keywords: Set[str]
    categories: List[str]
    suggested_action: str

    def to_row(self) -> str:  # Markdown row helper
        return (
            f"| {self.file} | {','.join(self.categories)} | {self.size} | {self.prints} | "
            f"{','.join(sorted(self.keywords))} | {self.suggested_action} |"
        )

    def to_json(self) -> dict:  # JSON-safe serialization (sets -> sorted lists)
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
    parser = argparse.ArgumentParser(description="Audit legacy test files for migration planning")
    parser.add_argument(
        "--markdown",
        dest="markdown_path",
        help="Optional path to also write the markdown summary table",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy_files = discover_legacy_tests()
    infos = [classify(p) for p in legacy_files]
    REPORTS_DIR.mkdir(exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump([i.to_json() for i in infos], f, indent=2)
    md = emit_markdown(infos)
    print("# Test Migration Audit\n")
    print(f"Repository Root: {REPO_ROOT}")
    print(f"Legacy root-level tests detected: {len(infos)}\n")
    print(md)
    print("\n(JSON written to reports/test_migration_audit.json)")
    if args.markdown_path:
        md_path = Path(args.markdown_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        print(f"(Markdown table also written to {md_path})")


if __name__ == "__main__":  # pragma: no cover
    main()
