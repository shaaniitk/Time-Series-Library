# Phase 1 Root Test Migration Mapping

This document records the migration of high-value root-level test scripts into the structured
`tests/modular` hierarchy. Redundant or superseded scripts will be slated for removal
once modular equivalents are stable.

| Root File | Classification | Modular Replacement | Action | Notes |
|-----------|----------------|---------------------|--------|-------|
| test_attention_components.py | component_smoke | tests/modular/components/test_attention_components_modular.py | KEEP (temp) | Remove after CI confirms new test passes consistently |
| test_decomposition_components.py | component_smoke | tests/modular/components/test_decomposition_components_modular.py | KEEP (temp) | Wavelet coverage retained |
| test_algorithmic_sophistication.py | algorithmic_depth (heavy) | tests/modular/algorithmic/test_algorithmic_attention_sophistication.py | KEEP (orig for now) | Heavy heuristics trimmed for runtime; original kept until parity confirmed |
| test_core_components.py | mixed (loss+attention integration) | (planned split: tests/modular/components/test_loss_components_modular.py & integration smoke) | PENDING | Will extract only essential loss + one attention integration |
| test_bayesian_fix.py | legacy_targeted_fix | (covered by probabilistic smokes) | CANDIDATE REMOVE | Validate feature overlap first |

## Next Steps
1. Implement modular loss component smoke extracted from `test_core_components.py`.
2. Add minimal integration test (loss + attention + backward) if not already covered by existing modular smokes.
3. Run focused pytest selection to validate new files: `pytest -q tests/modular/components tests/modular/algorithmic`.
4. If green across 3 consecutive runs, delete migrated root scripts above.
5. Update this mapping and proceed to remaining root tests.

## Removal Criteria
A root test can be removed when:
- All unique assertions / behaviors have modular equivalents.
- No bespoke debug printouts required for ongoing restoration.
- Runtime savings justify consolidation.

## Pending Inventory
Files not yet processed in Phase 1 mapping: core integration, bayesian fix, broad HF migration/chronos tests.
