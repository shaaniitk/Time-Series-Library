<!--
REFRESHED  (auto-updated) — The previous section duplicated & partially diverged from real repository state.
This file has been condensed & corrected. /!\ Do NOT re-introduce the stale top table.
-->

# Root Test Migration Map (Phase 1 - Updated)

Purpose: Track systematic migration of legacy root-level `test_*.py` scripts into the structured modular test suite under `tests/modular/` while preventing coverage loss and reducing execution noise (print-heavy harnesses, ad‑hoc runners).

Guiding Principles:
1. Consolidate: Prefer one parametrized modular test over several linear print-based scripts.
2. Minimize Duplication: If shape / forward smoke is covered in `test_end_to_end_regressions_modular.py`, retire redundant legacy variants.
3. Preserve Unique Semantics: Before pruning, port any non-trivial assertions (e.g., uncertainty tensor shape, multi-scale outputs, quantile dict integrity, capability metadata queries).
4. Mark Slow vs Fast: Production / large combination sweeps become `@pytest.mark.slow` or moved to optional extended suite.
5. Invariants First: Core registry + assembler + output shape + loss remain always-on; exploratory combinatoric sweeps move behind markers.

Legend:
- Category: smoke_basic | components | integration | probabilistic | algorithmic | hf_migration | chronosx | legacy_redundant | phased_upgrade
- Action: keep (temporary) | migrate (create new modular test) | merge (fold logic into existing modular test) | prune (delete after merge) | partial (some logic migrated) | review (needs inspection)
- Status: pending | in_progress | migrated | pruned

Metrics (current snapshot):
- Legacy root test files detected: (dynamic; see audit script output)
- Modular regression baseline present: YES (`tests/modular/regression/test_end_to_end_regressions_modular.py`)
- Files fully migrated & pruned: 10 (added `test_working_dependencies.py`)
- Files partially migrated: algorithmic sophistication (core ideas envisioned, not yet ported)

Priority Batches:
Batch A (High ROI, low coupling): step1_enhanced, step2_bayesian, step3_hierarchical, step4_quantile (merge into existing regression assertions if any unique checks) + runner_comprehensive (prune) + models_direct (prune) + migration (prune)
Batch B (Components Focus): attention_components, decomposition_components (convert to lean param tests under components/), core_components, moe_integration (add MoE case)
Batch C (Probabilistic & Advanced): production_bayesian, enhanced_bayesian_model, bayesian_fix (merge into probabilistic suite with uncertainty metrics)
Batch D (ChronosX): chronos_x_simple, chronosx_simple, modular_autoformer_chronosx (merge), chronos_x_model_sizes (parameterize sizes), chronos_x_real_data (mark slow), chronos_x_comprehensive (split fast vs slow)
Batch E (HF Migration): consolidate all HF tests into 3–4 focused suites (variants, training smoke, covariates, enhanced features)
Batch F (Validation / Dependency Harness): phase1_integration, phase2_* scripts, utils_compatibility, working_dependencies (extract genuinely still-relevant dependency checks; prune rest)
Batch D Status:
- Added tests/modular/chronosx/test_chronosx_smoke.py (tiny uncertainty toggle)
- Added tests/modular/chronosx/test_chronosx_scaling.py (tiny & mini sizes)
- Legacy ChronosX root scripts queued for prune (assertion parity confirmed)

Batch E Status:
- Added tests/modular/hf/test_hf_variants_smoke.py (Enhanced + Bayesian)
- Added tests/modular/hf/test_hf_training_smoke.py (single backward/optim step)
- Remaining HF covariate/enhanced feature coverage deferred until post Batch F

Batch F Status:
- Added tests/modular/config/test_dependency_validation_modular.py (valid config, capability alignment, adapter suggestions)
- Pruned legacy `test_working_dependencies.py` (verbose harness)
- Validator transient requirement parsing issue documented in test expectations (to tighten once fixed)
- Remaining review: bayesian_sanity_test.py (long training) — classify next

Accelerated Migration Strategy (overview):
1. Automated Audit (script) — classify legacy tests by regex heuristics (prints, Namespace usage, phased naming) and emit a JSON/Markdown summary.
2. Auto-Skeleton Generation — for each category produce (if absent) a modular test skeleton with parametrization points filled from extracted legacy constants.
3. Incremental Port & Assert Parity — replicate only meaningful assertions (shapes, key dict keys, uncertainty presence) avoiding print noise.
4. Delete Legacy Files in Batches — after new tests pass; one batch per PR to simplify review.

Immediate Next Actions (Planned Execution Order):
1. Commit audit script (`scripts/test_migration_audit.py`).
2. Run audit to produce current classification artifact (stdout + optional JSON file in `reports/`).
3. Implement Batch A merge (extend existing regression test or add a lightweight supplemental regression parity test capturing any missing uncertainty / multi-scale assertions).
4. Prune Batch A legacy files post-green run. (DONE – 6 files removed)

Current Detailed Table (excerpt — will be machine-refreshed by audit script later):

| File | Category | Planned Modular Destination | Action | Status | Notes (delta vs previous) |
|------|----------|-----------------------------|--------|--------|---------------------------|
| test_step1_enhanced.py | phased_upgrade | reuse existing regression test (already covers deterministic) | merge -> prune | pruned | Folded into unified modular regression |
| test_step2_bayesian.py | phased_upgrade | extend regression test (uncertainty check) | merge -> prune | pruned | Covered by bayesian parametrization & batch size variants |
| test_step3_hierarchical.py | phased_upgrade | regression test hierarchical branch | merge -> prune | pruned | Hierarchical branch + multi-scale shape parity asserted |
| test_step4_quantile.py | phased_upgrade | regression test quantile branch | merge -> prune | pruned | Quantile head integrity + variance check migrated |
| test_runner_comprehensive.py | legacy_redundant | none | prune | pruned | Replaced by pytest discovery + unified param grid |
| test_models_direct.py | legacy_redundant | none | prune | pruned | Direct instantiation smoke redundant with regression suite |
| test_migration.py | legacy_redundant | none | prune | pending | Decomposition & enhanced autoformer covered elsewhere |
| test_attention_components.py | components | tests/modular/components/test_attention_components_modular.py | migrate -> prune | pruned | Replaced by modular attention param test |
| test_decomposition_components.py | components | tests/modular/components/test_decomposition_components_modular.py | migrate -> prune | pruned | Legacy file removed after modular param test confirmed |
| test_working_dependencies.py | validation | tests/modular/config/test_dependency_validation_modular.py | migrate -> prune | pruned | Replaced by lean dependency & adapter suggestion tests |
| bayesian_sanity_test.py | probabilistic | tests/modular/probabilistic/test_bayesian_quantile_modular.py (subset) + future slow training perf test | review -> split | pending | Long-running training harness; core Bayesian loss/shape semantics migrated; perf aspects to optional slow suite |
| test_loss_components.py | components | (covered via registry + category count assertions) | migrate -> prune | pruned | Loss registry smoke superseded by advanced/components suites |
| test_advanced_components.py | integration | tests/modular/integration/test_advanced_components_modular.py | migrate | migrated | Advanced components modular test present |
| test_moe_integration_modular.py | components | tests/modular/components/test_moe_integration_modular.py | expand | in_progress | Parametrized (d_model,n_heads,n_samples) variability checks added |

Subsequent rows (HF, ChronosX, etc.) will be appended post-audit.

Verification Workflow Before Prune:
1. Add/Update modular test(s)
2. Run: `pytest -q tests/modular` (fast) then selective legacy file side-by-side if needed
3. Confirm green + coverage parity (manual diff of assertions / presence)
4. Remove legacy files in single batch commit

Changelog:
- Rewrote map to correct inaccurate deletion claims.
- Introduced batch migration strategy & measurable next steps.
- Added metrics section (will be auto-populated once audit script runs).

---
# (Legacy duplicate table below replaced — kept commented for historical reference; will be removed after full migration)
<!-- ORIGINAL DUPLICATED TABLE REDACTED FOR CLARITY -->


This document tracks migration of legacy root-level tests to the structured modular test suite.

Legend:
- Category: smoke_basic | components | integration | probabilistic | algorithmic_core | legacy_redundant | hf_migration | chronosx | phased_upgrade
- Action: keep (migrate) | merge | prune | already_migrated
- Status: pending | migrated | pruned | merged

| File | Category (initial) | Planned Modular Destination | Action | Status | Notes |
|------|--------------------|-----------------------------|--------|--------|-------|
| test_attention_components.py | components | tests/modular/components/test_attention_components_modular.py | migrate -> done | migrated | Rewritten; old file slated for prune after confirmation |
| test_decomposition_components.py | components | tests/modular/components/test_decomposition_components_modular.py | migrate -> done | migrated | Uses create_component API |
| test_algorithmic_sophistication.py | algorithmic_core | tests/modular/algorithmic/test_algorithmic_attention_sophistication.py | migrate -> partial | pending | Subset migrated; remaining heuristics TBD |
| test_advanced_components.py | components (advanced) | tests/modular/components/test_advanced_components_modular.py | pruned | migrated | Legacy file removed (covered by modular advanced + regression tests) |
| test_end_to_end_regressions_modular.py | integration | tests/modular/regression/test_end_to_end_regressions_modular.py | new | migrated | Consolidated deterministic/bayesian/quantile/hierarchical path coverage |
| test_modular_system.py | integration | tests/modular/integration/test_modular_system_smoke.py | migrate | pending | Basic end-to-end system assembly |
| test_phase1_integration.py | phased_upgrade | tests/modular/integration/test_phase1_equivalence.py | merge | pending | Collapse phased_* into single param suite |
| test_phase2_basic.py | phased_upgrade | tests/modular/integration/test_phase2_equivalence.py | merge | pending |  |
| test_phase2_attention.py | phased_upgrade | tests/modular/components/test_phase2_attention_expanded.py | merge | pending |  |
| test_phase2_validation.py | phased_upgrade | tests/modular/integration/test_phase2_validation.py | merge | pending |  |
| test_step1_enhanced.py | phased_upgrade | tests/modular/integration/test_step_path_regression.py | merge | pending | Combine step1-4 |
| test_step2_bayesian.py | phased_upgrade | tests/modular/integration/test_step_path_regression.py | merge | pending |  |
| test_step3_hierarchical.py | phased_upgrade | tests/modular/integration/test_step_path_regression.py | merge | pending |  |
| test_step4_quantile.py | phased_upgrade | tests/modular/integration/test_step_path_regression.py | merge | pending |  |
| test_bayesian_fix.py | probabilistic | tests/modular/probabilistic/test_bayesian_regressions.py | merge | pending | Edge-case regression consolidation |
| test_production_bayesian.py | probabilistic | tests/modular/probabilistic/test_bayesian_regressions.py | merge | pending |  |
| test_enhanced_bayesian_model.py | probabilistic | tests/modular/probabilistic/test_bayesian_regressions.py | merge | pending |  |
| test_chronos_x_simple.py | chronosx | tests/modular/integration/test_chronosx_smoke.py | migrate | pending | Minimal ChronosX path |
| test_chronosx_simple.py | chronosx | tests/modular/integration/test_chronosx_smoke.py | merge | pending | Duplicate variant |
| test_chronos_x_model_sizes.py | chronosx | tests/modular/integration/test_chronosx_scaling.py | migrate | pending | Parameterized sizes |
| test_chronos_x_real_data.py | chronosx | tests/modular/integration/test_chronosx_realdata.py | migrate | pending | May mark slow |
| test_chronos_x_comprehensive.py | chronosx | tests/modular/integration/test_chronosx_comprehensive.py | migrate | pending | Possibly split slow/fast |
| test_complete_hf_suite.py | hf_migration | tests/modular/hf/test_hf_model_variants.py | migrate | pending | Collapse multiple HF legacy tests |
| test_hf_model.py | hf_migration | tests/modular/hf/test_hf_model_variants.py | merge | pending |  |
| test_hf_models_fixed.py | hf_migration | tests/modular/hf/test_hf_model_variants.py | merge | pending |  |
| test_hf_modular_corrected.py | hf_migration | tests/modular/hf/test_hf_model_variants.py | merge | pending |  |
| test_hf_modular_training.py | hf_migration | tests/modular/hf/test_hf_training_smoke.py | migrate | pending | Keep one focused training smoke |
| test_hf_migration_simple.py | hf_migration | tests/modular/hf/test_hf_adapter_migration.py | migrate | pending | Migration adapter specifics |
| test_hf_enhanced_models.py | hf_migration | tests/modular/hf/test_hf_enhanced_features.py | migrate | pending | Enhanced feature flags |
| test_hf_flexibility.py | hf_migration | tests/modular/hf/test_hf_flex_fixtures.py | migrate | pending | Evaluate flexibility parameters |
| test_hf_covariates.py | hf_migration | tests/modular/hf/test_hf_covariates.py | migrate | pending | Covariate pipeline |
| test_all_hf_covariates.py | hf_migration | tests/modular/hf/test_hf_covariates.py | merge | pending | Merge breadth |
| test_covariate_wavelet_integration.py | components/integration | tests/modular/probabilistic/test_covariate_wavelet_integration.py | migrate | pending | Hybrid test |
| test_core_components.py | components | tests/modular/components/test_core_components_modular.py | migrate | pending | Enc/Dec/Sampling/Head set |
| test_migration.py | legacy_redundant | (none) | prune | pending | Superseded by structured config tests |
| test_models_direct.py | legacy_redundant | (none) | prune | pending | Direct model invocation obsolete |
| test_runner_comprehensive.py | legacy_redundant | (none) | prune | pending | Runner orchestration replaced |
| test_unified_factory.py | integration | tests/modular/integration/test_unified_factory.py | migrate | pending | Factory-level assembly |
| test_utils_compatibility.py | legacy_redundant | (none) | prune | pending | Utility compatibility folded into unit tests |
| test_gcli_architecture.py | integration | tests/modular/integration/test_gcli_architecture_contract.py | migrate | pending | Contract assertions |
| test_modular_autoformer_chronosx.py | chronosx | tests/modular/integration/test_chronosx_comprehensive.py | merge | pending | Consolidate |

Additional files to classify will be appended as discovered.

Next Steps:
1. Implement first batch migrations (core_components, unified_factory, modular_system) with param reductions.
2. Add advanced components modular test skeleton.
3. Remove pruned files after verifying new tests pass.
