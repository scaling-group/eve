# Workspace Notes

Current phase: Phase 2 solver and optimizer optimization.

{editable_files_block}
{editable_folders_block}

## Your Task

Treat `guidance/` as the current supporting guidance.

Your primary goal is to produce an improved solver candidate in `output/`. You should also improve the files in `guidance/` to distill your experience.

In this document, all paths are relative to the current phase workspace root unless stated otherwise. `output/` is the solver submission tree. `guidance/` is the optimizer-guidance tree for this phase workspace. `logs/optimize/` is a sibling optimization log directory under the same phase workspace root.

**MANDATORY:** Before you stop, you MUST invoke the predefined `check-runner` sub-agent from `.claude/agents/check-runner.md` or `.codex/agents/check-runner.toml`, depending on which runtime you are using. Have it execute its configured check workflow from the workspace root. The check-runner will provide necessary sanity checks without formal evaluation. Do NOT finish without running this check. If anything fails, repair the output and rerun the check until it passes.

Write any important optimization notes, intermediate artifacts, or debugging material directly into `logs/optimize/` while you work. The main program will preserve that directory as collected run logs and save your final response there.

Do not ask the human for clarification, approval, or feedback at any point during this run. Do the work autonomously, finish your edits, provide your final summary, and stop.

`output/` is pre-filled with the files from one of the reference examples. You may modify any editable files/folders within `output/`. You may also modify optimizer files inside `guidance/`. When you are satisfied, the editable files in `output/` will be extracted as the solver candidate submission, changed files in `guidance/` may be extracted as a new optimizer candidate, and `logs/optimize/` will be preserved separately as run logs.

## Reference Solver Examples

You have a few reference example(s). Their score cards are shown below:

{solver_examples_block}

{optimizer_examples_block}
