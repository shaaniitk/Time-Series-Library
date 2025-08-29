Split Plan (Updated with Analyzer Heuristics)

Purpose
Guide decomposition of oversized / multi-responsibility suites to align with modular architecture & layered CI (smoke / extended / perf / quarantine) while preserving unique assertion semantics.

Data Inputs
- Enhanced analyzer classifications (KEEP/EXTEND/PERF/SPLIT/DELETE)
- Line / assert density
- Heavy indicators (training loops, optimizer, backward, cuda)
- Semantic duplicate grouping (assert_signature)

Priority SPLIT Candidates (initial)
1. tests/test_modular_framework_comprehensive.py
   Axes: (a) registry + metadata; (b) component wiring/config resolution; (c) decomposition & attention component behaviors; (d) sampling & uncertainty; (e) validation/error cases; (f) performance/memory (move/mark perf)
2. tests/test_component_registry.py
   Axes: (a) singleton/structure; (b) registration & overwrite warnings; (c) config expansion/resolution; (d) validation errors; (e) migration template transformations
3. tests/test_loss_functionality.py (if present / large)
   Axes: separate each loss family: base (MSE/MAE) / quantile / frequency / probabilistic / calibration; move overlap to unit losses.
4. tests/test_integration.py
   Axes: (a) model forward shape parity; (b) training convergence micro; (c) quantile mode config coupling; (d) comparative architectures.
5. tests/test_series_decomposition.py
   Axes: (a) moving average & reconstruction; (b) trend vs seasonal quality metrics; (c) gradient flow & learnable weights; (d) edge cases (constant/short signals); (e) performance aspects (mark perf if timing enforced).
6. tests/test_performance_benchmarks.py
   Action: carve out pure timing/scaling assertions into dedicated perf/ subfiles by feature (scaling, parallelism, cache reuse) reducing monolith size.

Segmentation Guidelines
- Each new split file < 300 lines target; prefer thematic cohesion + single primary responsibility.
- Extract shared synthetic data / config builders into helpers (tests/helpers or existing integration/helpers).
- Minimize duplication of assert patterns—reuse helper assertion utilities where patterns repeat across splits (reduces semantic duplication and drift).
- Introduce local fixtures for repetitive construction tasks; keep smoke tests fixture-light for speed.

Marker Strategy Post-Split
- Core interface & shape: @pytest.mark.smoke
- Rich functional behavior (non-slow): @pytest.mark.extended
- Timing / scaling: @pytest.mark.perf
- Flaky thresholds still validating algorithm but unstable: @pytest.mark.quarantine

Planned Helper Modules
- tests/helpers/registry_builders.py: factory functions for registry and component configs.
- tests/helpers/data_generation.py: unified synthetic series/time-feature generators.
- tests/helpers/assertions.py: common assertion helpers (shape, finite, gradient presence, decomposition consistency heuristics).

Action Queue (Next Iterations)
1. Implement helper module skeletons.
2. Split test_modular_framework_comprehensive.py first (registry + wiring + decomposition). Retain original as thin orchestrator referencing new suites until stable, then deprecate.
3. Split test_component_registry.py similarly; move migration template assertions into migration-focused suite if overlap.
4. Relabel original large files with comment banner: DECOMPOSITION IN PROGRESS - DO NOT ADD NEW TESTS HERE.
5. After stable splits, remove or mark old monoliths as deprecated and update CONSOLIDATION_MANIFEST.md.

Metrics for Completion
- No SPLIT-classified file > 400 lines remaining.
- Total perf-marked tests isolated from extended runs.
- Smoke layer executes in < 30s on reference machine (target—adjust after measurement).
- Removal of duplicate assert_signature groups between newly split files.

Generated: 2025-08-09 (auto-updated)