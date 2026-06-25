# README

## Evolutionary Ensemble (EvE): Core Concepts

EvE maintains two coupled populations:

- `solver`: code or other files that directly work on the downstream task repo.
  The agent edits a solver candidate inside `solver/`, and that candidate is
  then evaluated on the concrete task.
- `optimizer`: markdown/toml guidance files that improve solvers, including
  prompts, notes, workflows, skill files, subagent files, etc. They are applied
  into a workspace to help the solver-optimization agent.

The loop runs in three phases:

1. Phase 1 samples reference optimizers and reference solvers from the current
   populations.
2. Phase 2 applies each sampled optimizer to sampled solver examples to produce
   a new solver candidate, and optionally a revised optimizer in the same step,
   then evaluates that candidate on the task.
3. Phase 3 updates optimizer scores from the relative performance of the solver
   candidates produced in Phase 2.

## Workspace

In this document, all paths are relative to the current workspace root unless
stated otherwise.

You should strictly work in the current workspace, which looks like the following.

```text
workspace_root/
├── AGENTS.md        ← workspace agent instructions
├── CLAUDE.md        ← workspace agent instructions
├── README.md        ← workspace-specific notes, must read.
├── guidance/        ← optimizer guidance files.
│   ├── skills/      ← skill tree exposed through `.claude/skills`
│   │                   and `.codex/skills`
│   ├── agents/      ← subagent trees exposed through `.claude/agents`
│   │   ├── claude/      and `.codex/agents`
│   │   └── codex/
│   └── other files  ← other optimizer guidance notes/docs, please read.
├── .claude/
│   ├── agents -> ../guidance/agents/claude
│   └── skills -> ../guidance/skills
├── .codex/
│   ├── agents -> ../guidance/agents/codex
│   └── skills -> ../guidance/skills
├── solver_examples/ ← sampled reference solvers from the population
│   └── <solver_id>/
│       ├── solver/   ← files from that solver example
│       ├── logs/     ← logs for that solver example
│       │   └── evaluate/     ← only evaluation logs are kept here
│       └── score.yaml ← evaluation score for that example
├── guidance_examples/ ← sampled reference optimizers
│   └── <optimizer_id>/
│       ├── guidance/ ← that optimizer's guidance files
│       ├── logs/      ← optimizer logs copied from storage
│       └── score.yaml ← current optimizer score
├── solver/          ← downstream task repo; editable files are prefilled.
├── logs/
│   └── optimize/    ← free-form log tree from this Phase 2 optimization run
```

You can modify the following parts in this workspace. Other files are read-only.

1. solver

You may modify any editable files/folders within `solver/`. Other files inside
`solver/` are read-only.

{editable_files_block}
{editable_folders_block}

2. optimizer guidance files

Optimizer guidance lives in the `guidance/` folder.

Guidance exposure:
- `guidance/skills/` is symlinked as `.claude/skills` and `.codex/skills`.
- `guidance/agents/claude/` is symlinked as `.claude/agents`.
- `guidance/agents/codex/` is symlinked as `.codex/agents`.

New or edited guidance skills and agents are visible through those symlinks
during the current and future EvE iterations.

The whole `guidance/` folder is editable, including ordinary guidance files,
`guidance/skills/`, and `guidance/agents/`, except the following read-only
immutable overlay files:

{immutable_overlay_block}

Use existing guidance skills and subagents when they help. When you learn
something reusable, update the most appropriate part of `guidance/`: ordinary
notes/docs for general strategy, `guidance/skills/` for reusable workflows or
skills, and `guidance/agents/` for specialist roles. These skills and agents
will be exposed through the provider symlinks.

3. optimization log

You can write ad hoc optimization notes, intermediate artifacts, or debugging
material into `logs/optimize/` while you work. The main program preserves that
directory separately as run logs.

## Your Task

Current phase: Phase 2 solver and optimizer optimization.

You have two goals:

1. Work hard to produce an improved Circle Packing solver candidate in
   `solver/`, using the guidance files, skills, and subagents in `guidance/` as
   active support rather than background context. Higher solver score is better.
2. Improve files in `guidance/` when your work reveals reusable guidance that
   would help the current and future iterations.

**MANDATORY:** Before you stop, you MUST invoke the predefined `check-runner`
sub-agent from `.claude/agents/check-runner.md` or
`.codex/agents/check-runner.toml`, depending on which runtime you are using.
Have it execute its configured check workflow from the workspace root. The
check-runner will provide necessary sanity checks without formal evaluation. Do
NOT finish without running this check. If anything fails, repair the output and
rerun the check until it passes.

Do not ask the human for clarification, approval, or feedback at any point during
this run. Do the work autonomously, finish your edits, provide your final
summary, and stop.

## Reference Solver Examples

Before editing, read the reference solver examples together with their
`logs/evaluate/` outputs. Do not rely on `score.yaml` alone; use evaluation logs
to identify rewarded packing moves, penalized failure modes, and reusable repair
strategies.

Also read the sampled optimizer examples, especially their `guidance/` files, to
identify reusable strategies and guidance improvements.

You have a few reference example(s). Their score cards are shown below:

{solver_examples_block}
{optimizer_examples_block}

## Score Semantics

Higher solver score is better.
