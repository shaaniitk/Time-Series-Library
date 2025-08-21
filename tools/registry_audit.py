"""Registry & Component Duplication Audit

This script scans the codebase for registry classes and component declarations
related to modular architecture (encoders, decoders, attention, decomposition,
losses, output heads, sampling) to aid consolidation into a unified registry.

Usage (from repo root):
    python -m tools.registry_audit

It produces a structured report to stdout listing:
  * Registries discovered (class name -> file)
  * Factory/helper functions (get_*_component patterns)
  * ComponentType enum members actually referenced in code
  * Duplicate class name occurrences across files

This is a non-intrusive developer aid and can be removed once consolidation
completes.
"""
from __future__ import annotations

import os
import re
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = [
    REPO_ROOT / "layers",
    REPO_ROOT / "utils",
    REPO_ROOT / "configs",
]

REGISTRY_PATTERN = re.compile(r"class\s+(\w+Registry)\b")
FACTORY_PATTERN = re.compile(r"def\s+(get_[a-zA-Z0-9_]*component)\b")
COMPONENT_CLASS_PATTERN = re.compile(r"class\s+([A-Z][A-Za-z0-9_]+)\((?:[A-Za-z0-9_,\s]*?)\):")
COMPONENT_TYPE_REF_PATTERN = re.compile(r"ComponentType\.([A-Z0-9_]+)")

EXCLUDE_DIR_KEYWORDS = {"__pycache__"}
PY_EXT = ".py"


def iter_python_files():
    for base in TARGET_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if any(part in EXCLUDE_DIR_KEYWORDS for part in path.parts):
                continue
            yield path


def scan_files():
    registries = {}
    factories = defaultdict(list)
    component_classes = defaultdict(list)
    component_type_refs = set()

    for file_path in iter_python_files():
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for m in REGISTRY_PATTERN.finditer(text):
            registries[m.group(1)] = str(file_path.relative_to(REPO_ROOT))

        for m in FACTORY_PATTERN.finditer(text):
            factories[m.group(1)].append(str(file_path.relative_to(REPO_ROOT)))

        for m in COMPONENT_CLASS_PATTERN.finditer(text):
            component_classes[m.group(1)].append(str(file_path.relative_to(REPO_ROOT)))

        for m in COMPONENT_TYPE_REF_PATTERN.finditer(text):
            component_type_refs.add(m.group(1))

    return registries, factories, component_classes, sorted(component_type_refs)


def build_report():
    registries, factories, component_classes, component_type_refs = scan_files()

    duplicates = {name: locations for name, locations in component_classes.items() if len(locations) > 1}

    report = {
        "registries": registries,
        "factory_functions": factories,
        "duplicate_component_class_names": duplicates,
        "component_type_enum_refs_found": component_type_refs,
        "summary": {
            "total_registries": len(registries),
            "total_factory_functions": len(factories),
            "duplicate_class_count": len(duplicates),
            "unique_component_type_refs": len(component_type_refs),
        },
    }
    return report


def main():
    report = build_report()
    print("=== REGISTRY & COMPONENT AUDIT REPORT ===")
    print(json.dumps(report, indent=2))
    print("=== END REPORT ===")

if __name__ == "__main__":
    main()
