# Evolutionary Ensemble (EvE) Overview

## Core Concepts

EvE maintains two coupled populations:

- `solver`: code or files that directly work on the downstream task repo. In
  Phase 2, the agent edits a solver candidate inside `output/` and that
  candidate is then evaluated on the concrete task.
- `optimizer`: markdown guidance files that improve solvers, including
  description, skills, suggestions, etc. They will be applied into a
  Phase 2 workspace to help the solver-optimization agent.

The loop runs in three phases:

1. Phase 1 samples reference optimizers and reference solvers from the current
   populations.
2. Phase 2 applies each sampled optimizer to sampled solver examples to produce
   a new solver candidate, and optionally a revised optimizer in the same step,
   then evaluates the solver candidate on the task.
3. Phase 3 updates optimizer scores from the relative performance of the solver
   candidates produced in Phase 2.

## Workspace

### Terminology

- `run root`: the outer Eve run directory for one whole experiment. It contains run-level artifacts such as databases, artifact stores, and all per-phase workspaces. Agents do not work directly in the run root.
- `phase workspace root`: the directory for one concrete Phase 2 agent run. When these instructions say `guidance/`, the reference example directories, `output/`, `logs/`, `README.md`, or `score.yaml`, they mean paths relative to the current phase workspace root.
- `output/`: the submission tree inside the current phase workspace root. Files under `output/` are the candidate result that will be extracted after the run.
- `logs/optimize/`: the optimization log directory inside the current phase workspace root. Use it for notes, intermediate artifacts, and debugging material that should be preserved as run logs, but should not become part of the candidate submission under `output/`.

### Solver Workspace (Phase 2)

```text
phase_workspace_root/
├── AGENTS.md        ← workspace agent instructions
├── CLAUDE.md        ← workspace agent instructions
├── README.md        ← workspace-specific notes, must read.
├── guidance/        ← optimizer guidance files copied from the selected optimizer
│   └── skills/      ← optional skill tree; exposed through `.claude/skills`
│                       and `.codex/skills`
├── .claude/
│   ├── agents/
│   │   └── check-runner.md   ← predefined Claude check sub-agent, copied from config
│   └── skills -> ../guidance/skills
├── .codex/
│   ├── agents/
│   │   └── check-runner.toml ← predefined Codex check sub-agent, copied from config
│   └── skills -> ../guidance/skills
├── solver_examples/ ← sampled reference solvers when
│                       `n_optimizer_examples_phase2 > 0`
│   └── <solver_id>/
│       ├── solver/   ← editable files from that solver example
│       ├── logs/     ← logs for that solver example
│       │   └── evaluate/     ← only evaluation logs are kept here
│       └── score.yaml ← evaluation score for that example
├── guidance_examples/ ← sampled reference optimizers when
│                         `n_optimizer_examples_phase2 > 0`
│   └── <optimizer_id>/
│       ├── optimizer/ ← optimizer files
│       ├── logs/      ← optimizer logs copied from storage
│       └── score.yaml ← current optimizer score
├── examples/        ← sampled reference solvers when
│                       `n_optimizer_examples_phase2 <= 0`
│   └── <solver_id>/
│       ├── solver/   ← editable files from that solver example
│       ├── logs/     ← logs for that solver example
│       │   └── evaluate/     ← only evaluation logs are kept here
│       └── score.yaml ← evaluation score for that example
├── output/          ← downstream task repo; editable files are prefilled
│                       from a solver in the active solver example directory.
│                       This is the submission tree.
├── logs/
│   ├── optimize/    ← free-form log tree from this Phase 2 optimization run
│   └── evaluate/    ← free-form log tree from evaluating the produced candidate
└── score.yaml       ← scores for the sampled solvers, the prefill solver, and the
                        produced solver
```

### Phase Workspace Lifecycle

Each phase workspace directory is named `<timestamp>_<id>` and is never deleted
after use. All phase workspaces accumulate on disk and can be inspected after
the fact.

```text
1. Build    fetch files and logs from database

2. Run      agent session executes in this directory

3. Extract  read output file(s) from the directory
            store as new entry in the database
```

If the agent exhausts its token budget without producing a valid output, step 3
is skipped and the run is recorded as failed in the database. The phase
workspace directory is retained either way.

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

## Score Semantics

Higher solver score is better.

=================VERY IMPORTANT RULE BELOW!!!=====================================
Don't keep improving forever. Once you think code is runnable and `check-runner` passed, STOP IMMEDIATELY, but only after one of these is true:
1. you made at least one research-bearing delta that is meaningfully distinct from the copied prefill solver, or
2. you explicitly recorded in `logs/optimize/session.md` why no such delta is justified in this workspace.
A validation-only repair such as checkpoint materialization hygiene does not by itself satisfy rule 1.
Never run evaluation in any sense by yourself, or perform a eval-edit loop. End the run and leave the evaluation and improvement to the future!
Before executing, repeat the above rule 3 times by yourself to make sure you have it in mind!!! Make sure you always follow it!!!
===================VERY IMPORTANT RULE ABOVE!!!===================================
