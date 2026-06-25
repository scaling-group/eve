---
name: inspect-population
description: "Use when reading what an Eve run evolved — its best solvers/optimizers, their code, and how the population improved"
---

# Inspect a run's population

Read what a run actually evolved: the best candidates, their code, and how scores progressed. Source of truth is the run's lineage DBs + artifacts (local; no wandb needed). run_root layout is in `docs/using-eve.md`.

## The candidates

Each run has `<run_root>/solver_lineage.db` and `<run_root>/optimizer_lineage.db`. The eve populations live in the `eve_population_entries` table, one row per candidate:

```bash
sqlite3 <run_root>/solver_lineage.db \
  "SELECT entry_id, created_at FROM eve_population_entries ORDER BY created_at;"
```

Swap in `optimizer_lineage.db` for optimizers. (Note: the DB also has generic `nodes` / `edges` tables, but the eve loop does not populate them, do not query those.)

## Scores (rank best-first)

Scores are not stored inline in the DB; each entry references a flat score artifact (`{status, score, summary}`) at `artifacts/<run-id>_solver/state/<...>__<entry_id>_score.yaml`. Rank candidates best-first:

```bash
for f in <run_root>/artifacts/*_solver/state/*_score.yaml; do
  s=$(awk -F': ' '/^score:/{print $2; exit}' "$f")
  printf '%s\t%s\n' "$s" "$(basename "$f")"
done | sort -rn | head
```

The filename carries the `<entry_id>`; top line = best so far. (The same flat score is also at each candidate's `<workspace>/logs/evaluate/score.yaml`. Do **not** parse the workspace-*root* `score.yaml`, that one is a different, nested phase-2 record, not the flat score.)

Smoke runs typically show `score: 0.0` with a `status: error` summary; that is the known circle_packing smoke payload error, not a skill problem.

## Read a candidate's evolved code

The agent's written solution is under that candidate's workspace: `<run_root>/solver_workspaces/<...>/solver/`. The archived copy is also stored inline as JSON in the entry's `files_ref_json` artifact under `<run_root>/artifacts/<run-id>_solver/state/`.

## Improvement over the run

Workspace directories are named `..._step_<n>_...`; compare the best `score.yaml` per step to see the trajectory. The optimizer log trees under `<run_root>/artifacts/<run-id>_optimizer/` show how the guidance evolved.

## Notes

- Completed-iteration count: `grep -c "Phase 3: updated Elo" <run_root>/runner.log`.
- To understand *why* a candidate behaved a certain way (not just its score/code), read its agent session: `../debug-agent-session/SKILL.md`.
