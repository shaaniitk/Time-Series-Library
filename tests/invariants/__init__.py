"""Invariant / algorithmic correctness test suite.

Separated from smoke / integration tests to allow focused development
and optional selective execution (e.g. with -k invariants).
"""

# Auto-load CI skip guard plugin if present
pytest_plugins = ["tests.invariants.ci_skip_guard"]
