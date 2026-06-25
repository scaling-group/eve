---
name: supervise-run
description: "Use when monitoring a running Eve experiment and recovering from problems while it runs"
---

# Supervise a run

Watch a live run, detect problems, and either recover (mechanical) or escalate (judgment). This skill is the monitoring loop; it delegates the actual fixes to the tool skills it references.

run_root layout, the completion marker, and data sources are in `docs/using-eve.md`.

## Monitoring loop

Run detached so you supervise from logs, not the foreground. (`codex_max` and `codex_smoke` run headless by default; the `open_iterm2: false` knob applies only to interactive/tmux-pane drivers and will error if passed to a subprocess driver.) On a sparse cadence (e.g. ~15 min steady-state, tighter right after launch or after a problem):

1. Read progress: completed iterations = `grep -c "Phase 3: updated Elo" <run_root>/runner.log`; confirm the lineage DBs were written recently.
2. Classify the run: running / stalled / halted / completed.
3. Act per the table.

## When to act vs escalate

Principle: **handle mechanical problems yourself; escalate scientific / taste judgments to the human.**

| Signal | Meaning | Action |
|---|---|---|
| `runner.log` tail shows a crash/halt, process gone | mechanical halt | recover: follow `../resume-run/SKILL.md` |
| no new `Phase 3` marker for a long time, process alive | stalled | inspect the agent's behavior via `../debug-agent-session/SKILL.md`; if truly stuck, pause + resume |
| persistent `synced 0 Phase 2 optimizers` **without** a benign "guidance tree was not modified" reason (see Notes) | silent no-op: optimizer guidance not landing | stop and flag to the human |
| score plateau or regression over many iterations | possibly a bad direction | flag to the human (do not silently change the experiment) |
| reached `loop.max_iterations` / clean exit | completed | report results (`../inspect-population/SKILL.md`) and stop |

Scientific calls — "is the optimizer bad?", "should the prompt be bolder?", "should we restart?" — are the human's. Surface them; act only when the human decides. They may then ask for an adjust (`../resume-run/SKILL.md`) or a fresh start keeping good solvers (`../import-run/SKILL.md`). The human can explicitly delegate authority (e.g. "you tune the prompt") — only then adjust autonomously.

## Notes

- Count completed iterations only from the `Phase 3` marker, never from `Iteration X / Y` lines (printed before completion).
- `synced 0 Phase 2 optimizers` just means no optimizer candidate was added that iteration. It is benign when paired with a `produce_optimizer_in_phase2 is enabled but the guidance tree was not modified` warning (the optimizer agent simply did not edit its guidance, normal in quick smokes). Only treat persistent `synced 0` as a real silent-no-op to flag in a longer run where you actually expect the optimizer to be evolving.
- Keep the runner process itself alive for the whole run (e.g. a persistent `tmux` session). A bare `nohup ... &` can leave half-initialized workspaces if the launching shell exits and the runner dies mid-iteration.
- Cluster / HPC submission and monitoring is environment-specific; if you run on a cluster, drive submission through your own cluster setup, not this skill.
