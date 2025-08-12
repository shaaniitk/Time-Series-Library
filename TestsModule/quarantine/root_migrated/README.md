# Root-Level Tests Migration (Quarantined)

These tests were originally located in the repository root as `test_*.py`.
To align with the modular testing framework and avoid duplicate/basename conflicts,
we relocated them under this quarantined folder.

By default, pytest will NOT collect tests in `TestsModule/quarantine/*` due to the
`norecursedirs` setting in `pytest.ini`.

## Why quarantine?
- Many of these tests target pre-migration structures or overlap with the modular suite.
- Some may import heavy or optional dependencies not needed for normal CI.
- This keeps our default pipelines lean while preserving history and content.

## How to run them explicitly
- Temporarily remove `TestsModule/quarantine` from `norecursedirs` in `pytest.ini`, or
- Run pytest pointing directly to a file or this directory:

```powershell
# Windows PowerShell
pytest TestsModule/quarantine/root_migrated -q
pytest TestsModule/quarantine/root_migrated/test_algorithmic_sophistication.py -q
```

## Next steps
- Gradually rework or de-duplicate these tests into the modular structure under `TestsModule/`.
- Prefer markers (smoke/extended/perf/quarantine/legacy) and shared fixtures from `conftest.py`.
