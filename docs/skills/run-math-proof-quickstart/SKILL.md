---
name: run-math-proof-quickstart
description: "Use when running a Math Proof quickstart from a user-provided problem: create a local task copy, launch a real EvE attempt, supervise it, and report proof/evaluation results."
---

Run this skill from the repository root. A Math Proof quickstart is a real local
attempt to solve the user's problem, not a smoke test.

Do not run Git commands as part of this skill. Do not `git add`, `git commit`,
`git push`, `git checkout`, or `git stash`. Quickstart application configs use
`application.path` so the runner can read local task files directly.

## Create The Task Pipeline

1. Choose a stable `<task_slug>` using lowercase letters, digits, and
   underscores. Derive `<task_slug_hyphen>` by replacing underscores with
   hyphens. For example, `infinitely_many_primes` becomes
   `infinitely-many-primes`. Quickstart copies are intentionally local scratch
   files, so place the task under `examples/tmp/`.
2. Refuse to overwrite an existing task unless the user explicitly asks. Check:

```bash
test ! -e examples/tmp/math_proof_<task_slug>
```

3. Copy the template task:

```bash
cp -R examples/math_proof/_quickstart examples/tmp/math_proof_<task_slug>
```

4. Replace `examples/tmp/math_proof_<task_slug>/problem/problem.md` with the
   user's problem. Keep `proof/.gitkeep`.
5. Check that the copied task files no longer contain template references:

```bash
rg "math_proof_quickstart|math-proof-quickstart|examples/math_proof/_quickstart|<task_slug>|<title>" \
  examples/tmp/math_proof_<task_slug>
```

Any remaining hit must be intentional. Fix stale template references before
launch. This generated quickstart task path is ignored by git.

## Launch

Run the quickstart config with task-specific overrides:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner \
  --config-name=math_proof_quickstart \
  application.name=math-proof-<task_slug_hyphen> \
  application.path=examples/tmp/math_proof_<task_slug> \
  label=math-proof-<task_slug_hyphen>
```

## Supervise And Report

After launch, supervise the live run with `supervise-run` skill. Use
`inspect-population` skill to read the completed population, evolved proof
files, and score artifacts.

For the final quickstart report, include:

- config name and exact launch command;
- run root;
- completed iteration count;
- whether the run completed, stalled, failed, or was interrupted;
- best candidate proof paths under `solver_workspaces/*/solver/proof/`;
- raw evaluation dimensions from score artifacts;
- evaluator, runner, boundary, or repair failure summaries;
- any useful next action, such as resume the same run or inspect a specific
  candidate proof.

Do not invent a scalar score for Math Proof. Preserve the vector dimensions and
state any presentation ordering as an operator view, not a stored score.
