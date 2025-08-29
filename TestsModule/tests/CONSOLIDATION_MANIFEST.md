# Test Consolidation Manifest (Updated Rationalization Steps 1–5 Draft)

## Phase 3A Deletions (Historical)
Redundant / monolithic root-level legacy tests emptied & documented.

## Phase 3B Scaffolds
Added focused smoke/perf/component/integration tests for core primitives and MoE metrics.

## Phase 3C Workflow Split
Monolithic `tests/integration/test_end_to_end_workflows.py` split into focused workflow tests:
- tests/integration/test_workflow_forecasting.py
- tests/integration/test_workflow_imputation.py
- tests/integration/test_workflow_anomaly_detection.py
- tests/integration/test_workflow_classification.py
- tests/integration/test_workflow_persistence.py
Shared helper:
- tests/integration/helpers/workflow_data.py
Original file replaced with stub comment.

## Rationalization Step 1 (Inventory Refresh)
- Legacy directory `tests_legacy/` empty (verified).
- Assertion density + file list captured via analyzer (to be extended with runtime sampling in future commit).

## Rationalization Step 2 (Preliminary Classification)
Classification taxonomy:
- KEEP: Focused, performant, unique semantic coverage.
- EXTEND (marker: extended): Broader functional suites not perf-heavy.
- PERF (marker: perf): Benchmark / scaling validation.
- QUARANTINE (marker: quarantine): Flaky / high-cost awaiting stabilization.
- SPLIT: Oversized multi-purpose tests queued for decomposition.
- DELETE: Redundant / superseded by focused coverage.

Initial classifications (high-level):
- KEEP (smoke): minimal_test.py, simple_test.py (consider trimming further), components/* small instantiation tests.
- EXTEND candidates: integration/component_validation/*, unit/components/* (retain richness but mark extended).
- PERF: test_performance_benchmarks.py, integration/performance_benchmarks, training_validation/test_performance_benchmarks.py
- SPLIT targets: test_modular_framework_comprehensive.py, test_component_registry.py (very large assertion count), test_loss_functionality.py (if duplicated with unit losses), test_integration.py (multi-purpose), test_series_decomposition.py (covers many modes).
- QUARANTINE candidates (need flakiness confirmation): performance scaling assertions with tight variance thresholds, caching speedup thresholds (subject to environment variance).
- DELETE candidates: overlapping quick scripts (run_* helpers) once organized workflows confirm equivalence; large deprecated runner variants (test_runner_comprehensive.py vs structured modular framework tests).

## Rationalization Step 3 (Relocation / Structure Plan)
- Ensure no test Python files remain at repo root (DONE previously).
- Consolidate any stragglers into categorized subdirs (in progress; pending scripted move of older top-level `tests/test_*` root files into deeper taxonomy or split/delete as above).
- Introduce `tests/quarantine/` directory for isolated unstable tests (to be created when first quarantine test identified).

## Rationalization Step 4 (Quarantine Mechanism)
- pytest.ini updated to add markers: extended, quarantine.
- Default CI recommendation: `pytest -m "not perf and not quarantine"`.
- Extended CI layer: include extended.
- Nightly layer: include perf; optionally quarantine with allow-fail.

## Rationalization Step 5 (CI Layering Guidance)
Suggested matrix:
1. smoke (fast PR): -m "smoke"
2. standard (PR required): -m "not perf and not quarantine"
3. extended (scheduled / label): -m "extended and not quarantine"
4. perf (nightly): -m "perf"
5. quarantine (manual / flaky watch): -m "quarantine" (allow-fail)

## Next Actions
- Implement runtime sampling & assertion signature hashing in analyzer.
- Tag files with appropriate markers or param-scope fixtures to reduce runtime.
- Split oversized suites per SPLIT targets list.
- Stand up quarantine directory when a flaky test is confirmed.
- Remove deprecated run_* orchestration scripts superseded by markers.

(Manifest auto-updated during Rationalization Phase Steps 1–5 draft.)