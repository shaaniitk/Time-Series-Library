# Financial Astrology TFT Plan Orchestrator

> This file tells an agent exactly how to resume, select, execute, verify, and
> hand off work without relying on chat history.

## 1. Mandatory Resume Protocol

Every session starts in this order:

1. Read `README.md`.
2. Read `CURRENT_STATUS.md`.
3. Read this file completely.
4. Read `PROGRESS_TRACKER.md`.
5. Read the active task card in `IMPLEMENTATION_PLAN.md`.
6. Read relevant hypothesis/data/feature specs.
7. Inspect repository and live processes read-only.
8. Confirm the task is `READY` and not owned.
9. Record ownership in the tracker before editing.

If instructions disagree, use this priority:

```text
current user instruction
CURRENT_STATUS.md explicit pause/scope
PROGRESS_TRACKER.md task state
IMPLEMENTATION_PLAN.md task card
supporting specifications
older repository-level plans
```

## 2. Current Orchestration Decision

```text
implementation: PAUSED
active external work: native TFT feature matrix
allowed project work: discussion, read-only inspection, documentation
next code task: none
```

Do not infer authorization from the existence of detailed task cards.

## 3. Read-Only Start Checks

Run:

```bash
pwd
git status --short
ps -eo pid,ppid,etime,stat,cmd | rg \
  'TFT_ETTh1_OT_feature_matrix|run.py|test_tft_deep_ett'
```

Then inspect only relevant paths:

```bash
rg --files projects/financial_astrology_tft
```

Rules:

- preserve the dirty worktree;
- do not reset or discard prior changes;
- do not signal or debug-attach the user's training process;
- do not edit scripts/code loaded by a live process unless the task explicitly
  owns that risk and the user has authorized it.

## 4. Task Selection Algorithm

Select exactly one primary task:

1. Is implementation paused? If yes, stop at discussion/read-only work.
2. Is an external run active? If yes, prefer independent documentation/data
   audit work and never modify its loaded code.
3. Find tasks whose dependencies are `COMPLETE`.
4. Exclude `BLOCKED`, `WAITING_EXTERNAL`, and already owned tasks.
5. Choose the earliest `READY` task in dependency order.
6. Read every referenced spec.
7. Record owner, timestamp, planned files, and acceptance commands.

Do not leap from `FA-DATA-001` directly to `FA-ENC-001`.

## 5. Claiming a Task

Before editing, update the tracker row:

```text
status: IN_PROGRESS
owner: <agent/session>
started: <timestamp>
planned_files:
acceptance_commands:
```

Add a short handoff note if another agent may work in parallel.

Only one agent owns a code file at a time.

## 6. File-Lock Groups

| Lock | Files |
|---|---|
| `LOCK-DATA` | `data_provider/*planet*`, data factory, dataset schema |
| `LOCK-FEATURE` | astrology feature/event/null builders |
| `LOCK-TFT-MODEL` | native TFT model and planetary encoder layers |
| `LOCK-EXP` | production experiment/training loop |
| `LOCK-CLI` | `run.py`, printed args, config/digest |
| `LOCK-TEST` | overlapping test modules |
| `LOCK-DOC` | this project tracker/orchestrator/registry |

Parallel work is safe only when locks do not overlap and dependencies genuinely
permit it.

## 7. Required Task Workflow

### 7.1 Before editing

1. Inspect relevant current implementation.
2. Search for existing utilities/tests with `rg`.
3. Record current focused-test result.
4. Check for user changes in the same files.
5. Reconfirm the task acceptance criteria.

### 7.2 Test-first contract

For code tasks:

1. add the smallest failing contract test;
2. confirm it fails for the intended reason;
3. implement the minimal slice;
4. run focused tests;
5. run the affected native TFT regression tests;
6. run diff hygiene.

Never add a broad architecture and only then decide what correctness means.

### 7.3 Verification

At minimum:

```bash
python -m pytest <focused tests> -q
git diff --check
```

For native TFT model changes, also run the existing core/extension/experiment
contracts selected by repository scope.

For data/features:

- prefix-invariance/no-future tests;
- timezone/holiday boundary tests;
- circular boundary tests;
- independent ephemeris/event fixtures;
- transform-fit boundary tests.

### 7.4 Finish

Before marking complete:

1. record exact commands and outcomes;
2. record artifact paths;
3. append experiment metadata if training occurred;
4. update decisions if semantics changed;
5. update the tracker atomically;
6. update `CURRENT_STATUS.md`;
7. update `SESSION_HANDOFF.md`;
8. release ownership.

If any acceptance item is missing, use `VERIFYING`, not `COMPLETE`.

## 8. Experiment Lifecycle

Every training run receives:

```text
experiment_id
timestamp
git_commit_or_worktree_digest
config_hash
data_hash
convention_hash
hypothesis_registry_hash
feature_spec_hash
fold_manifest_hash
seed
parameter_count
artifact_directory
status
metrics
```

Append one JSON object to `EXPERIMENT_REGISTRY.jsonl`. Never edit an old result
to match a new run; append a superseding entry.

Required artifacts:

```text
resolved config
ordered schema
fitted transforms
fold dates
checkpoint
per-date predictions
per-date losses
summary metrics
null manifest
interpretation diagnostics
stdout/stderr log
```

## 9. Hypothesis Change Control

Before results:

- a `DRAFT` rule may be edited with a decision-log entry;
- a confirmatory rule becomes `FROZEN-CONFIRMATORY` and receives a hash.

After results:

- changing a formula, orb, horizon, sign, anchor, or feature membership creates
  a new version;
- the old result remains in the scorecard;
- the new run is exploratory until independently confirmed.

Outer-planet features never enter a classical profile through a configuration
default.

## 10. TFT Matrix Closeout Procedure

`FA-TFT-001` is unusual because it runs in the user's terminal.

Allowed:

```bash
ps -eo pid,ppid,etime,stat,cmd | rg \
  'TFT_ETTh1_OT_feature_matrix|run.py'
find results -maxdepth 1 -type d -name \
  'long_term_forecast_tft_ot_p24_*' -print
```

After natural completion:

1. enumerate the 13 expected cases from the script;
2. verify each result directory and metric array;
3. compare prediction arrays where backend parity is expected;
4. load quantile diagnostics;
5. record failures separately;
6. update `RESULTS_SCORECARD.md`;
7. select a native reference by a declared rule;
8. mark `FA-TFT-001` complete.

Do not kill a slow case simply because later cases are queued.

## 11. Data Review Procedure

When the user supplies data:

1. copy nothing into permanent dataset paths until provenance is clear;
2. inspect shape, columns, sample timestamps, and missingness;
3. map each column to observed/target/known/static/forbidden;
4. reproduce selected ephemeris rows;
5. inspect Hilbert implementation;
6. draft the convention manifest;
7. show unresolved choices to the user before freezing them.

The output of data review is an audit and tests, not a trained model.

## 12. Stop Conditions

Stop and request direction when:

- the decision timestamp or target semantics cannot be inferred safely;
- source timestamps or units conflict;
- a proposed anchor time/location is materially uncertain;
- the only way forward would inspect a locked holdout;
- a new action expands scope beyond this project;
- the user asks to keep implementation paused.

Do not stop merely because a result is null. A well-run null result is a valid
project outcome.

## 13. End-of-Session Template

Update `SESSION_HANDOFF.md` with:

```text
date:
phase:
active_task:
implementation_authorized:
files_changed:
tests_run:
evidence_paths:
live_external_processes:
blockers:
next_safe_action:
```

The final user message should state:

- outcome first;
- what is actually complete;
- what remains paused/waiting;
- the most important evidence;
- the next decision or input.

