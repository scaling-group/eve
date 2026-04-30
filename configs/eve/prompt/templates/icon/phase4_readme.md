# Workspace Notes

Current phase: Phase 4 optimizer optimization.

## Your Task

Treat `guidance/` as supporting guidance, if it exists.

Your goal is to improve the optimizer guidance files in `output/`. These files are not executed in Phase 4 itself; they will later be copied into Phase 2 solver workspaces as supporting guidance for the solver-optimization agent.

The optimizer guidance you are editing lives in `output/docs/`. Start with `problem.md` for task context, then `directions.md` for the PE family menu, and `mutation_surface.md` for the editable file boundaries that future Phase 2 solvers will inherit.

Use the reference optimizers and the current `output/` files together to decide what guidance should be improved for the future Phase 2 solver workspace.

Use `examples/<optimizer_id>/logs/<solver_id>/` to inspect the historical solver results produced by each optimizer in past Phase 2 runs. Those historical solver results include solver files, optimize logs, evaluate logs, and final score.

Use `task_base/` to inspect the downstream task base. `task_base/` is the base working environment that a solver produced in Phase 2 will be filled into. Compare `task_base/` with `examples/<optimizer_id>/logs/<solver_id>/solver/` to see how a produced solver sits inside that task environment.

Write any important optimization notes, intermediate artifacts, or debugging material directly into `logs/optimize/` while you work. The main program will preserve that folder, append token usage metadata, and save your final response there.

`output/` is your working copy of one optimizer guidance set. It contains guidance files for a future Phase 2 solver workspace, not the target repo itself.

`output/` should normally contain only Markdown guidance files. These Markdown files will later be copied into the Phase 2 `guidance/` folder and read by the Phase 2 solver agent as supporting instructions for improving the solver. It is pre-filled with the files from one of the reference examples. You may add, remove, rename, or modify any files within `output/`. Keep `output/` limited to final optimizer guidance content only. Do not create logs, scratch files, reports, or other runtime artifacts under `output/` (for example `output/logs/`). Write all optimization notes and intermediate artifacts to `logs/optimize/` instead.

Do not expect target-repo `.py` files to exist here. Those files live in later Phase 2 solver workspaces; your job here is to write guidance for the agent that will see them there.

The live solver score is `-mean_d1_d10` over the full d1..d10 demo-count window. The optimizer-side score reflects how well solvers produced under your guidance performed on this metric. When you guide future Phase 2 solvers, write guidance that points them toward actual `mean_d1_d10` reduction without masking a degraded short-context regime.

Be careful with low-budget noise. If the visible examples were trained at a short step budget where `mean_d1_d4` and `mean_d1_d10` are both around the same large plateau (~0.4), score differences of 0.005-0.05 between candidates at that regime are mostly run-to-run noise, not optimization signal. Avoid promoting noise-level differences as evidence in your guidance updates.

In addition to the single-run noise floor, remember that each Phase 2 solver workspace sees only a small rank-sampled window of solver history. With a rank-softmax sampler favoring higher-scoring candidates, same-state negative re-runs are easy to filter out before a solver ever sees them. You usually have a wider view through `examples/<optimizer_id>/logs/<solver_id>/`, so watch for cases where the sampled window likely overstated a winner by hiding a contrary reproduction. When that happens, surface the counterexample directly in your next-round guidance so the solver is not reasoning from a survivor-filtered history.

Every actionable claim in the guidance you produce must be labeled either `visible-supported` or `inherited, not re-verified here`. `visible-supported` claims must cite a specific solver id and either the live `mean_d1_d10`, the diagnostic `mean_d1_d4`, or the raw `d1..d10` error curve from the sampled logs visible in this workspace. `inherited, not re-verified here` claims may be carried forward, but they must stay caveated and must not drive your highest-confidence recommendations. If you can upgrade an inherited claim to `visible-supported` from this workspace's evidence, do so explicitly; otherwise keep it downgraded or drop it from the high-confidence section.

There is intentionally no optimizer-side `check-runner` in this workspace. Do not try to run a remote smoke, solver-side validation sub-agent, boundary workflow, or task evaluation from Phase 4. Keep the work scoped to guidance files, write any working notes to `logs/optimize/`, and stop once the guidance is coherent.

Do not ask the human for clarification, approval, or feedback at any point during this run. Do the work autonomously, finish your edits, provide your final summary, and stop.

## Before halting (mandatory inspection)

Before you produce a final summary and stop, you **must** have read the actual contents of:

1. `guidance/` - the lead optimizer's proposed direction
2. `output/` - your working copy (prefilled from one reference optimizer)
3. At least one `examples/<optimizer_id>/logs/<solver_id>/` solver log - historical evidence

"Coherent guidance" is a judgment made **against** the content of `guidance/` and `output/`, not against the README alone. If after inspection you genuinely conclude there is no substantive improvement to make, record that conclusion explicitly in `logs/optimize/session.md` and cite which files you read. A silent halt (single-turn, README-only, zero edits) is not an acceptable outcome.

## Reference Optimizers

You have a few reference optimizer(s) in `examples/`. Read them to see examples of guidance files that were previously used for Phase 2 solver optimization. Their score cards are shown below:

{reference_optimizers_block}
