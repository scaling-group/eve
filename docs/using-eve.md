# Using EvE

Orientation for any agent (Claude Code or Codex) operating this repo. Included from `CLAUDE.md` / `AGENTS.md`.

## What EvE is

EvE is an evolutionary loop for optimizing solutions to a task. Worker agents ("solvers") propose and edit candidate solutions; a co-evolving "optimizer" shapes how they explore; each candidate is scored by your evaluation; results are tracked as a lineage so the population improves over iterations. You run it from the repo root with `uv` + Hydra configs.

## Your role

When you open this repo you are **operating** EvE: you set up a task, launch runs, then **supervise, pause, adjust, resume, import, inspect, and debug** them. The skills under `docs/skills/` are your verbs; we have linked `.agents/skills` and `.claude/skills` to this folder so supported agent entrypoints can load them automatically. Reach for the matching skill instead of improvising.

## Boundary (do not cross)

- Do not modify the EvE algorithm core (phase 2 / phase 3 / sampling / the optimizer algorithm).
- Operate through configs, the run lifecycle, and a run's artifacts, not by editing the engine.

## The run lifecycle (mental model)

set up task → launch (smoke first) → **supervise** → **pause / adjust / resume** → **import** (restart, keep good solvers) → **inspect** / **debug agent** → analyze.

## Skills (the toolbox)

- `supervise-run` — watch a live run; detect halts/stalls; recover or escalate. Delegates to the skills below.
- `configure-eve-driver` — choose between smoke/max driver presets and apply one-off interactive/debug overrides.
- `resume-run` — continue the **same** run after a pause/interruption, optionally after editing immutable/prompt/config.
- `import-run` — start a **new** run seeded from prior run(s): keep good solvers, swap in a fresh optimizer.
- `inspect-population` — read the evolved solver/optimizer code and how it improved across the lineage.
- `debug-agent-session` — read an agent's own rollout transcript to debug its behavior (stuck, gaming, bad reasoning).

Authoring a new task is covered by the existing skills: `implement-evaluation-steps`, `implement-subagent`, `implement-check-subagent`.

## Shared concepts (the skills refer back here)

- **run_root** — each run materializes at `.runs/eve/<app-name>/<run-id>/` containing `solver_workspaces/`, `evaluation_workspaces/`, `artifacts/<run-id>_solver|_optimizer/` (`transcripts/`, `state/`, `evaluations/`), `telemetry/`, `solver_lineage.db`, `optimizer_lineage.db`, `runner.log`, `checkpoint.json`, `.snapshots/`, and sometimes `.resume_archive/`.
- **Lineage DBs** — `solver_lineage.db` / `optimizer_lineage.db` hold the evolving populations in the `eve_population_entries` table (one row per candidate; scores are referenced score artifacts, not inline). Source of truth for progress and results.
- **Logical state vs operational trace** — `checkpoint.json`, `.snapshots/`, lineage DBs, and local `telemetry/` describe the logical experiment state. `runner.log`, agent transcripts, and `.resume_archive/` are operational traces useful for debugging.
- **Iteration-completion marker** — for quick operator checks, the `runner.log` line `Phase 3: updated Elo ... synced ... Phase 2 optimizers` marks a completed iteration. Do **not** count `Iteration X / Y` lines (printed at the start, before completion). For resume correctness, use `checkpoint.json`, lineage DBs, and `telemetry/` as the source of truth.
- **checkpoint + snapshots** — `checkpoint.json` records `last_completed_iteration`; `.snapshots/` holds per-iteration lineage-DB snapshots. Together they make a run resumable (rollback to the last completed iteration). On resume, local CSV telemetry is restored to the same anchor. Written automatically when `loop.enable_resume=true` (default).
- **resume vs import** — `resume` continues the **same** run (same id, same lineage). `import` starts a **new** run seeded from one or more prior runs' populations. They are mutually exclusive.

## Reading a run (data sources)

Read locally: the lineage DBs (table `eve_population_entries`), `runner.log`, `artifacts/`, and each step's `score.yaml`. **wandb is optional**: to run without it use `logger=many_loggers logger.wandb.enabled=false`; never assume wandb is configured.

## Command quick reference

These are bare commands; for the real procedures use the skills above.

- **Launch (smoke first):** `uv run python -m scaling_evolve.algorithms.eve.runner --config-name=<task>.smoke`, then `--config-name=<task>` for the full run. The repo ships `circle_packing` as the reference testbed.
- **Run detached:** `codex_max` and `codex_smoke` are headless by default. Only use `open_iterm2` when you intentionally switch to a tmux backend; see `docs/skills/configure-eve-driver/SKILL.md`.
- **Quick status:** `grep -c "Phase 3: updated Elo" <run_root>/runner.log` is a fast operational check; for the logical completed-iteration count, read `checkpoint.json` and `telemetry/iteration_metrics.csv`. For scores, rank the flat score artifacts `<run_root>/artifacts/*_solver/state/*_score.yaml` (one per entry, named `<entry_id>_score.yaml`) — best/latest selection (via `eve_population_entries.created_at`) is in `docs/skills/inspect-population/SKILL.md`.
- **Manage disk:** run roots grow (`solver_workspaces/`, `evaluation_workspaces/`, `artifacts/`, `.snapshots/`, `.resume_archive/`); prune or archive stale `.runs/eve/...` dirs you no longer need. Snapshot retention is config-controlled (see the `loop` config).
- **Common config knobs** (Hydra overrides; see the config files and `docs/skills/configure-eve-driver/SKILL.md` for the full driver surface): `loop.max_iterations`, `loop.n_workers_phase2`, driver `model` / `reasoning_effort`, `logger=many_loggers logger.wandb.enabled=false` (run without wandb).

## Code layout

For where the engine lives, see `docs/structure.md`.
