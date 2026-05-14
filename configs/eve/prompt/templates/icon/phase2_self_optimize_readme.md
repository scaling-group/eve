# Workspace Notes

Current phase: Phase 2 solver and optimizer optimization.

{editable_files_block}
{editable_folders_block}

## Your Task

Treat `guidance/` as the current supporting guidance.

Your primary goal is to produce an improved solver candidate in `output/`. You should also improve the files in `guidance/` to distill your experience.

In this document, all paths are relative to the current phase workspace root unless stated otherwise. `output/` is the solver submission tree. `guidance/` is the optimizer-guidance tree for this phase workspace. `logs/optimize/` is a sibling optimization log directory under the same phase workspace root.

The optimizer's guidance lives in `guidance/docs/`. Start with `problem.md` for task context, then `directions.md` for the PE family menu, and `mutation_surface.md` for editable file boundaries.

The live score is `score = -mean_d1_d10`, where `mean_d1_d10` is the mean question-QoI error across demo counts d1..d10. Higher score is better, because lower average error is better. `mean_d1_d4` remains a short-context diagnostic: a candidate that only looks better by distorting the early-demo regime is not a convincing result. Inspect the full `d1..d10` curve before declaring a candidate promising. If the `d1..d10` curve is roughly flat at a high plateau, the model is not yet trained; the score difference between candidates at that regime is mostly noise, not signal.

**MANDATORY:** Before you stop, you MUST invoke the predefined `check-runner` sub-agent from `.claude/agents/check-runner.md` or `.codex/agents/check-runner.toml`, depending on which runtime you are using. Have it execute its configured check workflow from the workspace root. Do NOT finish without running this check. If anything fails, repair the output and rerun the check until it passes.

The `check-runner` sub-agent is a gate, not a reward signal. Its contract is intentionally demanding: local syntax and boundary checks, a real remote remote cluster smoke, checkpoint confirmation, and a final settle pass. Do not let smoke-friendliness silently rank otherwise viable research mutations; if the only reason to choose the simpler path is that it looks easier to get through the smoke cycle, you are optimizing for the wrong thing. A mutation that fails smoke once is usually repairable, but an ambitious mutation you never try because its validation path looks riskier is permanently lost research signal.

If your workspace examples are dominated by one direction family, bias toward probing a non-dominant family from `guidance/docs/directions.md` rather than narrowly perturbing the dominant one again. Dominant-family perturbation is still acceptable when the evidence in your examples shows the dominant family has clear unexplored local structure you can target, but the default should favor diversification until at least one non-dominant family probe has been tested by a parallel worker. Record the family breakdown you observed in examples and your own choice (same-family perturbation vs non-dominant probe) explicitly in `logs/optimize/session.md`.

Write any important optimization notes, intermediate artifacts, or debugging material directly into `logs/optimize/` while you work. The main program will preserve that directory as collected run logs and save your final response there. If you substantially change the optimizer guidance, write a concise summary of the optimizer-side delta to `logs/optimize/optimizer_delta.md`.

Do not ask the human for clarification, approval, or feedback at any point during this run. Do the work autonomously, finish your edits, provide your final summary, and stop.

`output/` is pre-filled with the files from one of the reference examples. You may modify any editable files/folders within `output/`. You may also modify optimizer files inside `guidance/`. When you are satisfied, the editable files in `output/` will be extracted as the solver candidate submission, changed files in `guidance/` may be extracted as a new optimizer candidate, and `logs/optimize/` will be preserved separately as run logs.

Do not edit files outside `output/`, `guidance/`, and `logs/optimize/`.

Do not treat "prefill already passes" or "the safest move is no move" as sufficient by itself. Before you stop, either make at least one research-bearing delta that is meaningfully distinct from the copied prefill, or explicitly record why no such delta is justified in this workspace.

Validation-only hygiene can still matter, but it is not the research result. If you must repair a runner-facing detail such as checkpoint materialization to get a clean check, do that repair, but do not count that alone as the required research-bearing delta.

Guidance-side edits are part of the intended work in this mode. If your changes to `guidance/` improve the optimizer guidance tree, they may be extracted as a produced optimizer candidate and reused by later iterations.

## Reference Examples

Reference solver examples are in `solver_examples/` when optimizer examples are enabled, otherwise in `examples/`. Their score cards are shown below:

{solver_examples_block}

{optimizer_examples_block}
