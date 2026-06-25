---
name: debug-agent-session
description: "Use when an Eve agent is behaving wrong (stuck, gaming the score, bad reasoning) and you need to read its own rollout to find out why"
---

# Debug an agent's session

When a run looks wrong (stalled, gaming, no progress) and the score/population don't explain it, read the agent's own rollout, its full conversation and tool calls, to see *why* it did what it did. Local files only. run_root layout is in `docs/using-eve.md`.

## Where the rollout session is

Each active solver worker runs in `<run_root>/solver_workspaces/<workspace_id>/`. The agent's own session record (JSONL, one event per line) lives there, by driver:

- Codex: `<workspace_id>/.codex-driver-transcripts/<session>.jsonl` (full rollout also under `<workspace_id>/.codex-driver-home/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`)
- Claude Code: `<workspace_id>/.claude-driver-transcripts/<session>.jsonl`

A `*-live.jsonl` is an in-progress session; the non-live file is the completed snapshot. (Note: `<run_root>/artifacts/<run-id>_solver/transcripts/` holds the optimization *log trees*, not the agent conversation, use the workspace JSONL for behavior.) To find the workspace for a suspect candidate, locate its `..._step_<n>_...` directory under `<run_root>/solver_workspaces/` (see `../inspect-population/SKILL.md`); there is no DB column mapping an entry to its workspace path.

If the run has been resumed, failed post-checkpoint workspaces may have been moved to `<run_root>/.resume_archive/resume_*/solver_workspaces/<workspace_id>/`. Check active `solver_workspaces/` first for the current logical run, then `.resume_archive/` only when debugging the interrupted attempt itself.

## What to look for

Read the JSONL for the agent's reasoning + tool calls. Common failure signatures, with the `runner.log` markers that flag them:

- **Optimizer no-op** (`the guidance tree was not modified; no optimizer candidate will be produced`): the optimizer agent isn't editing its guidance. Read its session to see whether it understood the task.
- **Synced 0 optimizers** (`Phase 3: updated Elo for N optimizers and synced 0 Phase 2 optimizers`): the optimizer side produces nothing; two in a row is a stop-and-flag signal (see `../supervise-run/SKILL.md`).
- **Family lock-in:** `optimizer_lineage.db` shows only one optimizer `entry_id` reused across iterations (selection keeps picking the same one), exploration collapsed.
- **Gaming the eval:** the candidate scores high but its `solver/` (via `../inspect-population/SKILL.md`) games the metric instead of solving the task; the session often shows the agent reverse-engineering the evaluator.

## Note

A mechanical halt (crash, killed process) is not an agent-behavior problem, recover it with `../resume-run/SKILL.md`.
