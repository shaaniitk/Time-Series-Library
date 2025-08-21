# Modular Consolidation Refactor Plan (Tracked Memory)

This file records the ongoing consolidation of component registries and factories.

## Goals
- Single unified registry & factory for all modular components.
- Eliminate duplicate component class implementations across `configs/`, `layers/`, `utils/`.
- Centralize canonical component code under `layers/modular/` (attention, encoder, decoder, decomposition, sampling, output_heads, losses, fusion, adapters, backbones, modifiers).
- Deprecate and remove per-domain registries after shadow period.
- Simplify `configs/schemas.py` to rely on a single mapping from `ComponentType` to registry key.
- Safe deletion of `utils/modular_components` and legacy duplicates once tests pass.

## Phases
1. Audit (DONE): registry_audit.py generated duplication map.
2. Create unified interfaces + registry + factory (IN PROGRESS).
3. Shadow register canonical components (subset) while leaving old registries untouched.
4. Adapt assembler to prefer unified registry (fallback to legacy).
5. Expand registration to all component families.
6. Remove duplicate class definitions in `configs/concrete_components.py` that already exist in modular paths.
7. Deprecate old registries with thin wrappers logging warnings.
8. Delete obsolete directories (`utils/modular_components`, duplicated losses, sampling, etc.).
9. Final documentation & testing sweep.

## Decisions Pending
- Trend/seasonal handling abstraction (proposed `TrendProvider` capability flag).
- Location of Bayesian modifier pipeline (proposed `layers/modular/modifiers`).

## Open Questions (to review later)
- Which enhanced vs standard encoder variants to keep as canonical? (Proposed: keep modular versions, drop config-defined duplicates.)
- Retain hierarchical placeholder or replace with a NotImplemented shim until real implementation lands?

## Current Step
Implement unified `interfaces.py`, `registry.py`, `factory.py` under `layers/modular/core/` and register initial subset.

