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
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

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
    test_functions: int
    import_domains: int
    assert_count: int
    heavy_indicators: int  # heuristic count of expensive patterns


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


TEST_FUNC_PATTERN = re.compile(r'^def +test_[A-Za-z0-9_]+', re.MULTILINE)
IMPORT_PATTERN = re.compile(r'^(?:from +([\w\.]+) +import|import +([\w\.]+))', re.MULTILINE)


INTERNAL_ROOT_NAMES: Tuple[str, ...] = (
    'tests', 'chronos', 'autoformer', 'modular', 'components', 'core', 'models', 'TestsModule'
)


HEAVY_PATTERNS = [
    re.compile(r'for +epoch'),
    re.compile(r'optimizer\.'),
    re.compile(r'train\('),
    re.compile(r'fit\('),
    re.compile(r'backward\('),
    re.compile(r'cuda', re.IGNORECASE),
]

# Domain keyword heuristics (broad coverage of component & workflow concepts)
DOMAIN_KEYWORDS = [
    'backbone', 'moe', 'expert', 'routing', 'loss', 'quantile', 'workflow', 'registry',
    'decomposition', 'encoder', 'decoder', 'attention', 'embedding', 'adapter', 'output',
    'fusion', 'hierarchical', 'forecast', 'imputation', 'anomaly', 'classification', 'persistence'
]


def _extract_metrics(text: str) -> Tuple[int, int, int, int]:
    """Return (test_function_count, import_domain_count, assert_count, heavy_indicator_count)."""
    test_fn_count = len(TEST_FUNC_PATTERN.findall(text))
    imports: Set[str] = set()
    for m in IMPORT_PATTERN.finditer(text):
        mod = m.group(1) or m.group(2) or ''
        if not mod:
            continue
        top = mod.split('.')[0]
        if top in INTERNAL_ROOT_NAMES or top not in {
            'os', 'sys', 're', 'json', 'math', 'typing', 'pathlib', 'itertools', 'functools', 'collections', 'dataclasses'
        }:
            imports.add(top)
    assert_count = text.count('assert ')
    heavy_ind = sum(1 for pat in HEAVY_PATTERNS if pat.search(text))
    return test_fn_count, len(imports), assert_count, heavy_ind


VENV_DIR_NAMES = {'venv', '.venv', 'env', '.env', 'tsl-env', 'build', 'dist'}


def _skip_dir(dirpath: str) -> bool:
    parts = {p.lower() for p in dirpath.split(os.sep)}
    return not parts.isdisjoint(VENV_DIR_NAMES)


def collect_test_files(internal_only: bool = True) -> List[TestFileInfo]:
    out: List[TestFileInfo] = []
    for dirpath, _, files in os.walk(REPO_ROOT):
        if _skip_dir(dirpath):
            continue
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
                test_fn_count, import_domains, assert_count, heavy_ind = _extract_metrics(text)
                rel = os.path.relpath(full, REPO_ROOT)
                if internal_only and not (
                    rel.startswith('tests'+os.sep)
                    or rel.startswith('tests_legacy'+os.sep)
                    or rel.startswith('test_')
                    or rel.startswith('TestsModule'+os.sep)
                ):
                    continue
                out.append(TestFileInfo(full, rel, f, h, lines, infer_category(rel, f), test_fn_count, import_domains, assert_count, heavy_ind))
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
    """Run analysis and emit structured JSON plus split/classification guidance.

    Extended metrics:
      - runtime_sample_seconds: optional single-pass import & collection timing per file
      - assert_signature: hashed multiset of normalized assert lines (to detect semantic duplication)
      - classification: preliminary KEEP/EXTEND/PERF/SPLIT/DELETE/QUARANTINE suggestion
    """
    files = collect_test_files(internal_only=True)
    exact_dups, name_collisions, category_counts, large_scripts = analyze(files)

    # Lightweight runtime sampling: time import + file execution of top-level (no test run)
    import time
    runtime_samples: Dict[str, float] = {}
    for info in files:
        start = time.perf_counter()
        # Execute compile only (avoid running tests) for a safer timing proxy
        try:
            with open(info.path, 'r', encoding='utf-8', errors='ignore') as fh:
                src = fh.read()
            compile(src, info.path, 'exec')
        except Exception:
            pass
        runtime_samples[info.rel_path] = time.perf_counter() - start

    # Derive assert signatures for duplication detection (normalize whitespace & numeric literals)
    assert_signatures: Dict[str, str] = {}
    domain_hits: Dict[str, List[str]] = {}
    for info in files:
        try:
            with open(info.path, 'r', encoding='utf-8', errors='ignore') as fh:
                lines = fh.readlines()
        except OSError:
            continue
        normalized_asserts: List[str] = []
        text_blob_parts: List[str] = []
        for line in lines:
            if 'assert ' in line:
                norm = re.sub(r'\s+', ' ', line.strip())
                norm = re.sub(r'\b\d+\b', 'N', norm)
                # Remove file-specific variable names heuristically (simple pattern)
                norm = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)', lambda m: 'VAR' if m.group(1) not in {'assert', 'True', 'False'} else m.group(1), norm)
                normalized_asserts.append(norm)
            # Collect for domain scanning (lowercased)
            text_blob_parts.append(line.lower())
        sig = hashlib.md5('\n'.join(sorted(normalized_asserts)).encode('utf-8')).hexdigest() if normalized_asserts else ''
        assert_signatures[info.rel_path] = sig
        blob = ' '.join(text_blob_parts)
        hits = sorted({kw for kw in DOMAIN_KEYWORDS if kw in blob})
        domain_hits[info.rel_path] = hits

    # Preliminary classification heuristics
    classifications: Dict[str, str] = {}
    for info in files:
        cls = 'KEEP'
        if info.category == 'performance' or 'performance_benchmark' in info.rel_path:
            cls = 'PERF'
        elif info.lines >= 700 or (info.lines >= 500 and info.assert_count > 150):
            cls = 'SPLIT'
        elif info.heavy_indicators > 0 and info.lines > 400:
            cls = 'SPLIT'
        elif info.lines > 400:
            cls = 'EXTEND'
        elif info.assert_count > 250:
            cls = 'SPLIT'
        # Potential deletion: duplicate content hash or near-empty
        if info.lines < 30 and info.assert_count < 3 and info.test_functions <= 1:
            cls = 'DELETE'
        classifications[info.rel_path] = cls

    # Duplicate assert signature detection (semantic overlap)
    signature_groups: Dict[str, List[str]] = {}
    for rel, sig in assert_signatures.items():
        if not sig:
            continue
        signature_groups.setdefault(sig, []).append(rel)
    semantic_duplicates = [g for g in signature_groups.values() if len(g) > 1]

    # Augment file summaries with runtime + classification + domain keywords + suggested marker
    file_summaries = []
    def _suggest_marker(f: TestFileInfo) -> str:
        cls = classifications[f.rel_path]
        if cls == 'PERF':
            return 'perf'
        if cls == 'EXTEND':
            return 'extended'
        if cls == 'SPLIT':
            return ''  # will be split first
        if cls == 'DELETE':
            return ''
        # smoke heuristic: very small, few asserts, no heavy indicators
        if f.lines < 130 and f.assert_count <= 10 and f.heavy_indicators == 0 and f.test_functions <= 6:
            return 'smoke'
        return ''
    for f in files:
        file_summaries.append({
            'rel_path': f.rel_path,
            'lines': f.lines,
            'asserts': f.assert_count,
            'test_functions': f.test_functions,
            'import_domains': f.import_domains,
            'heavy_indicators': f.heavy_indicators,
            'category': f.category,
            'runtime_sample_seconds': round(runtime_samples.get(f.rel_path, 0.0), 5),
            'classification': classifications[f.rel_path],
            'assert_signature': assert_signatures.get(f.rel_path, ''),
            'domain_keywords': domain_hits.get(f.rel_path, []),
            'suggested_marker': _suggest_marker(f),
        })

    top_by_lines = sorted(files, key=lambda f: f.lines, reverse=True)[:15]
    top_by_tests = sorted(files, key=lambda f: f.test_functions, reverse=True)[:15]
    top_by_import_domains = sorted(files, key=lambda f: f.import_domains, reverse=True)[:15]
    top_by_asserts = sorted(files, key=lambda f: f.assert_count, reverse=True)[:15]

    report = {
        'summary': {
            'total_test_files': len(files),
            'exact_duplicate_groups': len(exact_dups),
            'name_collision_groups': len(name_collisions),
            'line_size_threshold': LINE_SIZE_THRESHOLD,
        },
        'category_counts': category_counts,
        'classification_counts': {c: list(classifications.values()).count(c) for c in sorted(set(classifications.values()))},
        'semantic_duplicate_groups': len(semantic_duplicates),
        'semantic_duplicates': semantic_duplicates[:25],
        'file_summaries': file_summaries,
        'scoring_notes': 'heavy_indicators>0 suggest slow/integration style test (training loops, optimizer, cuda, etc.)',
        'top_line_count': [
            {
                'rel_path': f.rel_path,
                'lines': f.lines,
                'test_functions': f.test_functions,
                'import_domains': f.import_domains,
                'asserts': f.assert_count,
                'heavy_indicators': f.heavy_indicators,
            } for f in top_by_lines
        ],
        'top_test_functions': [
            {
                'rel_path': f.rel_path,
                'test_functions': f.test_functions,
                'lines': f.lines,
                'import_domains': f.import_domains,
                'asserts': f.assert_count,
                'heavy_indicators': f.heavy_indicators,
            } for f in top_by_tests
        ],
        'top_import_domains': [
            {
                'rel_path': f.rel_path,
                'import_domains': f.import_domains,
                'lines': f.lines,
                'test_functions': f.test_functions,
                'asserts': f.assert_count,
                'heavy_indicators': f.heavy_indicators,
            } for f in top_by_import_domains
        ],
        'top_asserts': [
            {
                'rel_path': f.rel_path,
                'asserts': f.assert_count,
                'lines': f.lines,
                'test_functions': f.test_functions,
                'import_domains': f.import_domains,
                'heavy_indicators': f.heavy_indicators,
            } for f in top_by_asserts
        ],
        'large_scripts': [
            {
                'rel_path': f.rel_path,
                'lines': f.lines,
                'test_functions': f.test_functions,
                'import_domains': f.import_domains,
                'asserts': f.assert_count,
                'heavy_indicators': f.heavy_indicators,
            } for f in large_scripts
        ],
        'suggested_canonicalization': [
            {
                'keep': sorted(group, key=lambda i: (0 if i.rel_path.startswith('tests') else 1, len(i.rel_path)))[0].rel_path,
                'remove': [g.rel_path for g in sorted(group, key=lambda i: (0 if i.rel_path.startswith('tests') else 1, len(i.rel_path)))[1:]],
            }
            for group in exact_dups[:30]
        ],
    }

    print(json.dumps(report, indent=2))

    # Human guidance sections
    print("\nSPLIT_RECOMMENDATIONS")
    for f in top_by_lines[:20]:
        cls = classifications[f.rel_path]
        if cls == 'SPLIT':
            print(f" - {f.rel_path}: {f.lines} lines, asserts={f.assert_count}, heavy={f.heavy_indicators} -> SPLIT CANDIDATE")
    print("\nDELETE_CANDIDATES")
    for f in files:
        if classifications[f.rel_path] == 'DELETE':
            print(f" - {f.rel_path}: lines={f.lines}, asserts={f.assert_count}, tests={f.test_functions}")
    print("\nSEMANTIC_DUPLICATE_GROUPS (assert_signature based)")
    for group in semantic_duplicates[:10]:
        print(" - GROUP:")
        for rel in group:
            print(f"    * {rel}")
    print('\nDONE')


if __name__ == '__main__':
    main()
