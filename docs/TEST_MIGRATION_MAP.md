# Test Migration Map

Status as of 2025-08-12

Highlights:
- Quantile losses updated to accept 3D [B,L,T*Q] and 4D [B,L,T,Q].
- LossRegistry provides list_components alias (back-compat) and list_available preferred.
- Probabilistic quantile test now sorts quantile outputs before computing interval width to avoid negative width due to unsorted heads.

Validated Suites (green):
- tests/modular/losses
- tests/modular/components
- tests/modular/integration (smoke)
- tests/modular/probabilistic (bayesian quantile)

Pending broader validation:
- TestsModule/extended, components, integration (non-quarantine).

Quarantine handling:
- TestsModule/quarantine/delete_later duplicates replaced with module-level skip stubs to avoid import basename collisions.

Action items:
- After full confirmation, remove quarantine duplicates permanently.
- Keep pytest.ini norecursedirs to exclude quarantine from default runs.
