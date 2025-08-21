# Legacy Registry Deprecation Plan

This document formalizes the phased removal of legacy per-component registries
(attention, encoder, decoder, decomposition, sampling, output_head, loss, fusion)
now superseded by the unified modular registry.

## Phase 1 (Current)
- Legacy registry modules still exist.
- Each exposes its original API but methods are monkey‑patched to forward to
  `unified_registry` and emit a single `DeprecationWarning`.
- All components have been registered in the unified registry (see
  `MIGRATION_MAPPING.md`).
- Self‑test (`python -m tools.unified_registry_selftest`) passes for every
  registered component.

## Phase 2 (Transition – After Test Suite Migration)
- Update all internal tests to use `unified_registry.create` instead of
  `AttentionRegistry.get/ create` etc.
- Replace any fixture-level imports of legacy registries with unified ones.
- Introduce CI job that fails if legacy registry APIs are called (can be done
  by temporarily replacing monkey‑patched methods to raise RuntimeError unless
  an env var `ALLOW_LEGACY_REGISTRY` is set during transition).

## Phase 3 (Removal)
- Delete legacy registry modules.
- Provide minimal stub modules that raise a clear error:
  `ImportError("<Name>Registry removed – use unified_registry (see MIGRATION_MAPPING.md)")`.
- Remove deprecation docs and warnings.

## Guardrails
- Before entering Phase 3 run a repository‑wide grep to confirm no imports of
  `.attention.registry` etc. remain outside deprecated stubs.
- Re‑run full test matrix with warnings treated as errors to ensure nothing
  implicitly still touches legacy APIs.

## Rationale
A single registry drastically simplifies:
- Introspection & automatic documentation generation.
- Configuration validation (enum alignment with registrations).
- Avoidance of name collisions and drift across families.
- Future integration with HF / external tooling.

## Open Follow‑Ups
- Add lint rule (custom ruff plugin or pre-commit regex) to forbid legacy
  registry imports once Phase 2 completes.
- Consider JSON export helper of unified registry for dynamic UI layers.
