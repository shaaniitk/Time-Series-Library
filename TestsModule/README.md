# TestsModule Framework

This directory hosts the NEW modular test hierarchy. All *new* or migrated tests
must live under `TestsModule/`.

## Structure
- helpers/ : shared data generation, assertions, registry builders (to add)
- smoke/ : critical fast-path construction & forward tests (`@pytest.mark.smoke`)
- components/ : per-component functional tests (subfolders by domain)
- integration/ : cross-component & workflow tests
- extended/ : heavy edge-case & large horizon tests (`@pytest.mark.extended`)
- perf/ : performance guardrail tests (`@pytest.mark.perf`)
- quarantine/ : flaky or under-investigation (`@pytest.mark.quarantine`)
	- delete_later/ : physical holding area for Stage 3 DELETE-classified trivial placeholder tests (removed in final cleanup phase).
- meta/ : analyzer & registry metadata sanity tests

## Runners
- `python TestsModule/run_all.py` runs every test under this hierarchy.
- `python TestsModule/smoke/run_smoke.py` runs only smoke tests.
- `python TestsModule/integration/run_integration.py` integration layer.
- `python TestsModule/extended/run_extended.py` extended layer.
- `python TestsModule/perf/run_perf.py` performance layer.
- `python TestsModule/quarantine/run_quarantine.py` quarantine layer.
- `python TestsModule/meta/run_meta.py` meta tests.

You can also invoke directly with pytest: `pytest TestsModule -m smoke`.

## Adding a Test
1. Choose the directory (e.g., components/backbone/).
2. Name file `test_<area>_<behavior>.py`.
3. Use helpers from `helpers/` (avoid inline random data logic).
4. Add markers where needed (`@pytest.mark.smoke`, `@pytest.mark.extended`, etc.).
5. Keep each file single-responsibility.

## Migration Phases
Stage 0 baseline captured outside this tree. Migration will gradually move
root-level tests here with classification (KEEP/SPLIT/EXTEND/PERF/DELETE).

### Stage Artifacts
- internal_test_inventory_stage3.json : Analyzer output (root + tests + tests_legacy + TestsModule).
- split_plan_stage3.json : Planned splits for monolithic SPLIT-classified tests (generated via `python tools/generate_split_plan.py`).
- deletion_candidates_stage3.json : Manifest of trivial tests relocated into `quarantine/delete_later` (via `python tools/isolate_deletion_candidates.py`).

## Determinism
Global seed fixture enforces reproducibility. Override via `TS_TEST_SEED` env var.

## Maintenance Scripts
- tools/analyze_tests.py : Produces inventories & heuristic classifications.
- tools/generate_split_plan.py : Builds structured split plan for SPLIT candidates.
- tools/isolate_deletion_candidates.py : Relocates DELETE-classified trivial tests into quarantine/delete_later with manifest.
