"""Utility to analyze test file duplication and categorization.

Run directly: python tools/analyze_tests.py
Outputs summary JSON-like sections for:
  - exact_duplicates (same content hash)
  - name_collisions (same filename different content)
  - category_counts (by inferred category)
  - large_scripts (files exceeding line threshold)

Inferred categories based on path segments / filename keywords.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_PREFIX = 'test_'
LINE_SIZE_THRESHOLD = 400  # lines above which treat as candidate for benchmarks/examples


@dataclass
class TestFileInfo:
    path: str
    rel_path: str
    filename: str
    hash: str
    lines: int
    category: str


def infer_category(rel_path: str, filename: str) -> str:
    p_lower = rel_path.lower()
    if 'performance' in p_lower or 'benchmark' in p_lower:
        return 'performance'
    if 'integration' in p_lower or 'end_to_end' in p_lower or 'component_validation' in p_lower:
        return 'integration'
    if 'training_validation' in p_lower:
        return 'training'
    if 'unit' in p_lower:
        return 'unit'
    if 'chronosx' in p_lower:
        return 'chronosx'
    if 'core_algorithms' in p_lower:
        return 'core_algorithms'
    if 'modular_framework' in p_lower:
        return 'modular_framework'
    if 'utilities' in p_lower:
        return 'utilities'
    # Heuristic by filename keywords
    f = filename.lower()
    if 'autoformer' in f or 'bayesian' in f or 'chronos' in f:
        return 'model'
    if 'migration' in f:
        return 'migration'
    if 'registry' in f or 'component' in f:
        return 'registry'
    return 'misc'


def collect_test_files() -> List[TestFileInfo]:
    out: List[TestFileInfo] = []
    for dirpath, _, files in os.walk(REPO_ROOT):
        for f in files:
            if f.startswith(TEST_PREFIX) and f.endswith('.py'):
                full = os.path.join(dirpath, f)
                try:
                    with open(full, 'rb') as fh:
                        data = fh.read()
                except OSError:
                    continue
                h = hashlib.md5(data).hexdigest()
                try:
                    text = data.decode('utf-8', errors='ignore')
                except Exception:
                    text = ''
                lines = text.count('\n') + 1
                rel = os.path.relpath(full, REPO_ROOT)
                out.append(TestFileInfo(full, rel, f, h, lines, infer_category(rel, f)))
    return out


def analyze(files: List[TestFileInfo]):
    # Exact duplicates by hash
    hash_groups: Dict[str, List[TestFileInfo]] = {}
    for info in files:
        hash_groups.setdefault(info.hash, []).append(info)
    exact_duplicates = [g for g in hash_groups.values() if len(g) > 1]

    # Name collisions (same filename different hash)
    name_groups: Dict[str, Dict[str, TestFileInfo]] = {}
    for info in files:
        name_groups.setdefault(info.filename, {})[info.hash] = info
    name_collisions = {name: list(hmap.values()) for name, hmap in name_groups.items() if len(hmap) > 1}

    # Category counts
    category_counts: Dict[str, int] = {}
    for info in files:
        category_counts[info.category] = category_counts.get(info.category, 0) + 1

    large_scripts = [info for info in files if info.lines >= LINE_SIZE_THRESHOLD]

    return exact_duplicates, name_collisions, category_counts, large_scripts


def main():
    files = collect_test_files()
    exact_dups, name_collisions, category_counts, large_scripts = analyze(files)

    print('SUMMARY')
    print(' total_test_files:', len(files))
    print(' exact_duplicate_groups:', len(exact_dups))
    print(' name_collision_groups:', len(name_collisions))
    print('\nCATEGORY_COUNTS')
    for k, v in sorted(category_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f' {k}: {v}')

    print('\nEXACT_DUPLICATE_GROUPS (first 10)')
    for group in exact_dups[:10]:
        print(' - HASH', group[0].hash)
        for f in group:
            print('    ', f.rel_path)

    print('\nNAME_COLLISION_GROUPS (first 10)')
    for i, (name, entries) in enumerate(list(name_collisions.items())[:10]):
        print(' - FILE', name)
        for e in entries:
            print(f'    {e.rel_path}  (hash={e.hash[:8]})')

    print('\nLARGE_SCRIPTS (line_threshold=', LINE_SIZE_THRESHOLD, ') first 10')
    for info in large_scripts[:10]:
        print(f' - {info.rel_path} ({info.lines} lines)')

    # Suggested canonical path decisions (heuristic): prefer under tests/ directory
    print('\nSUGGESTED_CANONICALIZATION')
    for group in exact_dups[:30]:
        # sort to put tests/ paths first
        sorted_group = sorted(group, key=lambda i: (0 if i.rel_path.startswith('tests') else 1, len(i.rel_path)))
        keep = sorted_group[0]
        remove = [g for g in sorted_group[1:]]
        print(' keep:', keep.rel_path)
        for r in remove:
            print('  remove:', r.rel_path)

    print('\nDONE')


if __name__ == '__main__':
    main()
