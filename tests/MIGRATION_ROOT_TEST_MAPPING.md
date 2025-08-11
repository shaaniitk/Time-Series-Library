# Root Test Migration Mapping (Phase 1)

This document tracks migration of legacy root-level test scripts into the structured modular test suite.

Legend:
- keep_modular: Already covered or migrated into `tests/modular/*`.
- migrate_refactor: Needs refactor into new pytest-style modular test (create under appropriate subfolder).
- consolidate: Merge logic into an existing modular test (no new file).
- prune: Remove as redundant / superseded.

| Legacy File | Classification | Rationale | Action |
|-------------|---------------|-----------|--------|
| test_attention_components.py | consolidate | Functionally replaced by `tests/modular/components/test_attention_components_modular.py` using new registry API | prune after verifying no unique coverage (TODO verify advanced variants) |
| test_decomposition_components.py | consolidate | Replaced by `tests/modular/components/test_decomposition_components_modular.py` | prune |
| test_advanced_components.py | migrate_refactor | Mixes attention, decomp, encoder, decoder, sampling, loss enumerations; partial overlap but includes advanced enumerations not yet in modular tests | Split into: components/advanced/test_advanced_attention_enumeration.py & components/advanced/test_advanced_decomposition_wavelet.py (planned) |
| test_phase1_integration.py | migrate_refactor | Validates advanced loss classes & enum presence | Create `tests/modular/losses/test_advanced_losses_modular.py` |
| test_modular_system.py | consolidate | Registry creation & basic component instantiation now implicitly covered; minimal unique value | prune (retain key assertions moved into summary test) |
| test_working_dependencies.py | migrate_refactor | Demonstrates configuration validation pathways distinct from component forward tests | Create `tests/modular/config/test_configuration_validation.py` focusing on validator outputs |
| test_new_hf_models.py | prune | Script generator pattern; superseded by explicit modular tests | prune |

Next Steps:
1. Implement advanced loss modular test.
2. Implement configuration validation test.
3. Extract any unique assertions from advanced_components into focused advanced enumeration tests.
4. Delete pruned legacy scripts once replacements merged and passing.

Maintainer: Automated migration helper.
