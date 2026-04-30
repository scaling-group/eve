# Workspace Notes

Current phase: Phase 4 optimizer optimization.

## Your Task

Treat `guidance/` as supporting guidance, if it exists.

Your goal is to improve the optimizer guidance files in `output/`. These files are not for Phase 4 itself; they will later be copied into Phase 2 solver workspaces as supporting guidance for the solver-optimization agent.

In this document, all paths are relative to the current phase workspace root unless stated otherwise. `output/` is the submission tree. `logs/optimize/` is a sibling optimization log directory under the same phase workspace root, not a path inside `output/`.

Use the reference optimizers and the current `output/` files together to decide what guidance should be improved for the future Phase 2 solver workspace.

Use `examples/<optimizer_id>/logs/<solver_id>/` to inspect the historical solver results produced by each optimizer in past Phase 2 runs. Those historical solver results include solver files, optimize logs, evaluate logs, and final score.

Use `task_base/` to inspect the downstream task base. `task_base/` is the base working environment that a solver produced in Phase 2 will be filled into. Compare `task_base/` with `examples/<optimizer_id>/logs/<solver_id>/solver/` to see how a produced solver sits inside that task environment.

Write any important optimization notes, intermediate artifacts, or debugging material directly into `logs/optimize/` while you work. The main program will preserve that directory as collected run logs, append token usage metadata, and save your final response there.

`output/` is your working copy of one optimizer guidance set. It contains guidance files for a future Phase 2 solver workspace, not the target repo itself.

`output/` should normally contain only Markdown guidance files. These Markdown files will later be copied into the Phase 2 `guidance/` folder and read by the Phase 2 solver agent as supporting instructions for improving the solver. It is pre-filled with the files from one of the reference examples. You may add, remove, rename, or modify any files within `output/`. Keep `output/` limited to final optimizer guidance content only. Do not create logs, scratch files, reports, or other runtime artifacts under `output/` (for example `output/logs/`). Write all optimization notes to the sibling directory `logs/optimize/` instead so they are preserved as run logs rather than submission content.

Do not expect target-repo `.py` files to exist here. Those files live in later Phase 2 solver workspaces; your job here is to write guidance for the agent that will see them there.

Do not ask the human for clarification, approval, or feedback at any point during this run. Do the work autonomously, finish your edits, provide your final summary, and stop.

## Reference Optimizers

You have a few reference optimizer(s) in `examples/`.  Read them to see examples of guidance files that were previously used for Phase 2 solver optimization. Their score cards are shown below:

{reference_optimizers_block}
