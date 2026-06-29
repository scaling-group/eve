# Evaluation Workspace

This workspace evaluates one candidate proof. Work only in this evaluation
workspace. Treat `solver/problem/` and `solver/proof/` as read-only context.
Core's current evaluation workspace makes `solver/` read-only, so write the
formal evaluator submission under `logs/evaluate/evaluation/` and the raw score
card to `logs/evaluate/score.yaml`.

Workspace layout:

```text
.
|-- solver/
|   |-- problem/          # Read-only problem statement and reference material.
|   `-- proof/            # Read-only candidate proof submission.
|-- logs/
|   |-- optimize/         # Phase 2 solver-agent logs copied in as context.
|   `-- evaluate/         # Write evaluation/, score.yaml, checker reports here.
|-- solver_examples/      # Sampled reference solvers from the population.
|   `-- <solver_id>/
|       |-- solver/       # Files from that solver example.
|       |-- logs/         # Logs for that solver example.
|       |   `-- evaluate/ # Evaluation logs for that solver example.
|       `-- score.yaml    # Evaluation score for that solver example.
|-- .codex/agents/        # Codex evaluation helper agents.
|-- .claude/agents/       # Claude evaluation helper agents.
`-- ...
```

# Your Task

Read `solver/problem/` and proof files under `solver/proof/` only to coordinate
the scoring auditor subagents and write self-contained subagent tasks. Do not
rescore, reinterpret, or override the proof quality. Aggregate the scores given
by the auditors into the final `logs/evaluate/score.yaml`.

## Evaluation Artifacts

The step is complete when these artifacts exist:

```text
logs/evaluate/evaluation/coverage-auditor/     # Coverage review.
logs/evaluate/evaluation/correctness-auditor/  # Correctness review.
logs/evaluate/evaluation/dependency-auditor/   # Dependency review.
logs/evaluate/evaluation/clarity-auditor/      # Clarity review.
logs/evaluate/evaluation/strategy-auditor/     # Strategy review.
logs/evaluate/score.yaml                       # Final score card.
```

Write `logs/evaluate/score.yaml` by aggregating auditor
suggested scores. Scoring auditor subagents write their formal review files under
`logs/evaluate/evaluation/<auditor-name>/`.

Do not create a separate evaluator-authored critique. The formal review tree is
the set of auditor-written dimension directories under `logs/evaluate/evaluation/`.

## Scoring Auditors

For every nontrivial candidate evaluation, use the dedicated scoring auditor
subagents:

- `coverage-auditor`: theorem obligations and missing, weakened, or replaced
  coverage; decomposes the target theorem into obligations.
- `correctness-auditor`: validity of the written proof units; decomposes the
  proof into reasoning units.
- `dependency-auditor`: stated dependencies, external facts, and applicability;
  extracts named and implicit dependencies.
- `clarity-auditor`: organization, notation, local goals, and auditability;
  identifies the proof structure.
- `strategy-auditor`: proof-route quality, missed structural theorems,
  avoidable brute-force work, and route-level improvement targets.

Each auditor focuses on one dimension and returns evidence plus a suggested
score. Launch all scoring auditor subagents in parallel.

Do not write the final score card before the auditor review directories under
`logs/evaluate/evaluation/` have been written.

Invoke evaluator helper agents as fresh subagents of their declared type. Do
not use full-history fork mode such as Codex `spawn_agent` with `fork_context`.
Fresh subagents do not inherit your conversation context, so give each one a
self-contained task message.

## Aggregation Contract

Do not rescore the proof or invent dimension values in this evaluator. Build
the final score card by aggregating the scoring auditors'
suggested dimension scores into `logs/evaluate/score.yaml`. Use reference
material in `solver/problem/` only as optional context; the candidate must stand
on the proof written under `solver/proof/`.

The score file must use the math-proof dimensions-only score schema:

```yaml
dimensions:
  coverage: 88.0
  correctness: 82.0
  dependency: 76.0
  clarity: 83.0
  strategy: 55.0
```

`logs/evaluate/score.yaml` must contain exactly this `dimensions` mapping. The
five dimension values are numbers in `[0, 100]`. Do not write a headline
aggregate `score`.

## Score Schema Checker

Before you stop, invoke the predefined `score-schema-checker` subagent from
`.claude/agents/score-schema-checker.md` or
`.codex/agents/score-schema-checker.toml`, depending on the current runtime.
Run it after writing `logs/evaluate/score.yaml`. Do not finish until you have
reviewed that report and, if anything fails, repaired `score.yaml` and rerun
the checker.

Start a fresh `score-schema-checker` subagent from the current evaluation
workspace root. The local agent definition is a provider-native subagent, not a
shell command and not a tool named `score-schema-checker`. For Codex
`spawn_agent`, do not use full-history fork mode; start a fresh subagent and
give it a self-contained message that asks it to check
`logs/evaluate/score.yaml`.

The checker validates both the score schema and required evaluation artifacts,
including that `logs/evaluate/evaluation/` exists with at least one non-empty
file. Use the checker report only to fix score schema, YAML, numeric type,
metric-name, or missing evaluator-submission issues. The checker must not
reassess proof quality, and you remain responsible for the final contents of
`logs/evaluate/score.yaml` and `logs/evaluate/evaluation/`.
