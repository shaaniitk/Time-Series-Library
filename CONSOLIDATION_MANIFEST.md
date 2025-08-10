# Test Consolidation Manifest

## Phase 2 (Completed Earlier)
- Duplicate ChronosX tests removed (details previously documented)

## Phase 3A (Current) - Deletions of Low-Value Root Tests
Deleted empty or redundant broad-scope root tests:
- test_debug_cpu.py (empty)
- test_gcli_simple.py (empty)
- test_fft_isolated.py (empty)
- test_auxiliary_loss_fix.py (empty)
- test_complete_modular_framework.py (empty placeholder)
- test_comprehensive_modularization.py (monolithic 718 lines superseded by planned focused splits)
- test_complete_framework.py (broad integration superseded by targeted component + integration suites)

Rationale:
- Empty files provide no coverage.
- Monolithic comprehensive scripts duplicate forthcoming focused suites (registry/component/integration/perf) and impede fast feedback.
- Retained necessary coverage elements to be reintroduced via new split-focused tests (pending Phase 3B scaffolding):
  - Component registry tests
  - Focused attention & loss component smoke tests
  - MoE auxiliary metrics test
  - Lightweight integration (model build + forward) test
  - Performance/perf marker isolated tests

## Upcoming Phases
### Phase 3B - Scaffold Focused Replacement Suites
Planned new test files:
- tests/components/test_attention_components.py
- tests/components/test_loss_components.py
- tests/components/test_moe_aux_metrics.py
- tests/integration/test_model_build_and_forward.py
- tests/perf/test_attention_perf.py (marked @pytest.mark.perf)

### Phase 3C - Split Large Legacy Files
Targets: test_end_to_end_workflows.py, test_component_registry.py, test_modular_framework_comprehensive.py

### Phase 4 - Coverage & Gap Analysis
Add missing edge-case tests and ensure parametrized coverage without bloat.

---
Manifest updated after Phase 3A deletions.
