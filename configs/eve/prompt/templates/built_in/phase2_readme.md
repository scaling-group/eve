# Workspace Notes

Current phase: Phase 2 solver optimization.

{editable_files_block}
{editable_folders_block}

## Your Task

Treat `guidance/` as supporting guidance, if it exists.

Your goal is to produce an improved solver candidate guided by the optimizer files and any supporting materials in this phase workspace.

In this document, all paths are relative to the current phase workspace root unless stated otherwise. `output/` is the submission tree. `logs/optimize/` is a sibling optimization log directory under the same phase workspace root, not a path inside `output/`.

Before you stop, invoke the predefined `check-runner` sub-agent from `.claude/agents/check-runner.md` or `.codex/agents/check-runner.toml`, depending on which runtime you are using. Have it execute its configured check workflow from the repository root in `output/`, check-runner will provide necessary sanity check without formal evaluation. Do not finish until you have reviewed that report and, if anything fails, repaired the output and rerun the check.

Write any important optimization notes, intermediate artifacts, or debugging material directly into `logs/optimize/` while you work. The main program will preserve that directory as collected run logs and save your final response there.

Do not ask the human for clarification, approval, or feedback at any point during this run. Do the work autonomously, finish your edits, provide your final summary, and stop.

`output/` is pre-filled with the files from one of the reference examples. You may modify any editable files/folders within `output/`. You may also write notes under `logs/optimize/`. Do not modify other paths in the phase workspace. When you are satisfied, the editable files in `output/` will be extracted as your candidate submission, while `logs/optimize/` will be preserved separately as run logs.

## Reference Solver Examples

You have a few reference example(s). Their score cards are shown below:

{solver_examples_block}

{optimizer_examples_block}
