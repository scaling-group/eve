---
name: resume-run
description: "Use when continuing an interrupted or paused Eve run from where it stopped (same run, same id), optionally after editing the experiment"
---

# Resume a run

`resume` continues the **same** run (same run id, same lineage) from the last completed iteration. To instead start a **new** run seeded from one or more prior runs, see `../import-run/SKILL.md`.

run_root layout and the checkpoint/snapshot model are described in `docs/using-eve.md`.

## 1. Pause (if the run is live)

Kill the process at any time — `Ctrl-C`, `SIGTERM`, or `tmux kill-session`. **Do not wait for a "safe" boundary.** Each completed iteration is snapshotted, so an interrupted iteration is recoverable; the cleanup happens on resume, not on kill.

## 2. Adjust (optional, before resuming)

These take effect on resume (the run re-reads them from disk each iteration):

- the optimizer's `immutable/` and `prompt/` files
- the run config

**Does NOT take effect on resume:** editing `initial_guidance/`. The optimizer population is restored from the run's lineage DB, not re-seeded. To change the optimizer itself, start a fresh run instead — see `../import-run/SKILL.md`.

## 3. Resume

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=<config> resume_from=<absolute_run_root>
```

The runner does the recovery for you: it reads `<run_root>/checkpoint.json` for the last completed iteration, rolls the solver/optimizer lineage DBs back to that iteration's snapshot, restores local CSV telemetry to that same iteration, archives interrupted post-checkpoint workspaces under `<run_root>/.resume_archive/`, and continues under the **same run id**.

- `resume_from` must be an **absolute** path; a relative path is rejected.
- If you launched without wandb, re-pass `logger.wandb.enabled=false` here too (logger overrides are not persisted across the resume invocation).
- Optional: `resume_iteration=<K>` rolls back to an earlier completed iteration `K` (must be `<= last_completed_iteration`).
- Local telemetry under `<run_root>/telemetry/` is logical experiment output. After resume, it should contain rows through the checkpoint anchor plus newly completed post-resume rows, not rows from the failed partial attempt.
- `.resume_archive/resume_*/manifest.json` records any active post-anchor workspace paths that were moved out of the logical run surface. Use it when debugging the failed interrupted attempt.

## 4. Verify (quick sanity)

- An edit you made in step 2 shows up in the **post-resume** worker session logs/transcripts (the optimize session under the new workspace), and is **absent** from already-completed ones. Prompt edits are reflected in the worker session, not copied verbatim into the workspace as a file.
- The run continues under the original run id. Use `checkpoint.json`, lineage DBs, and `telemetry/` for logical state; `runner.log` is an operational trace and may include failed pre-resume attempt output.
- To read the resulting population, see `../inspect-population/SKILL.md`.

## Gotchas

- Resume needs snapshots. They are written every iteration when `loop.enable_resume=true` (the default). A run launched with `loop.enable_resume=false` has no snapshots and **cannot** be resumed.
- Do not delete `<run_root>/.snapshots/` or `checkpoint.json` — a missing snapshot makes resume fail hard.
- `resume_from` and `import_from` are mutually exclusive; setting both is an error.
- Remote wandb state is outside the local rollback contract. If wandb is enabled, treat it as remote operational logging, not as the source of truth for same-run resume.
