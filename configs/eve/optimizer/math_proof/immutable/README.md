# README

## Evolutionary Ensemble (EvE): Core Concepts

EvE maintains two coupled populations:

- `solver`: code, math proof, or other files that directly work on the downstream task repo.
  The agent edits a solver candidate inside `solver/` and that
  candidate is then evaluated on the concrete task.
- `optimizer`: markdown/toml guidance files that improve solvers, including prompts, notes,
  workflows, skill files, subagent files, etc. They are applied into a
  workspace to help the solver-optimization agent.

The loop runs in three phases:

1. Phase 1 samples reference optimizers and reference solvers from the current
   populations.
2. Phase 2 applies each sampled optimizer to sampled solver examples to produce
   a new solver candidate, and optionally a revised optimizer in the same step,
   then evaluates that candidate on the task.
3. Phase 3 updates optimizer scores from the relative performance of the solver
   candidates produced in Phase 2.

## Workspace

In this document, all paths are relative to the current workspace root unless stated otherwise.

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
│   └── other files  ← ordinary optimizer guidance notes/docs, must read all files.
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
│       │   └── evaluate/     ← evaluation logs for that solver example
│       └── score.yaml ← evaluation score for that solver example
├── guidance_examples/ ← sampled reference optimizers
│   └── <optimizer_id>/
│       ├── guidance/ ← guidance files in that optimizer example
│       ├── logs/      ← history logs for that optimizer example
│       └── score.yaml ← latest score for that optimizer example
├── solver/          ← downstream task repo; editable files are prefilled.
│   └── proof/       ← proof and all proof-supporting dependency files.
└── logs/optimize/  ← ad hoc Phase 2 logs only; not part of the proof evidence chain
```

You can modify the following parts in this workspace (others are read-only):

1. solver

You may modify any editable files/folders within `solver/`. Other files inside `solver/` are read-only.

Keep the complete proof evidence chain inside `solver/proof/`.
Any file needed to state, check, support, or reproduce the
proof, including helper code or data files, generated lemmas, runnable checking
scripts, and verification scripts, belongs in `solver/proof/`, not in
`logs/optimize/`.

{editable_files_block}
{editable_folders_block}

2. optimizer guidance files

Optimizer guidance are in `guidance/` folder.

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

Use existing guidance skills and subagents when they help. 
When you learn something reusable, update the most appropriate part of `guidance/`: ordinary
notes/docs for general strategy, `guidance/skills/` for reusable workflows or
skills, and `guidance/agents/` for specialist roles. These skills and agents
will be exposed through the provider symlinks. Feel free to add new files.

Do not limit guidance updates to task facts. Some reusable guidance should be
self-prompts: any instructions that change how the agent works, including
working style, workflow, or guidance update principles themselves.

3. optimization log:

You can write ad hoc optimization notes, intermediate artifacts, or debugging material
into `logs/optimize/` while you work, which will be preserved separately as run logs.
Do not rely on `logs/optimize/` as proof evidence; evaluation should be able to
use the proof and its dependencies from `solver/proof/`.

## Your Task

Current phase: Phase 2 solver and optimizer optimization.


You have two goals:


1, Produce a stronger proof candidate in `solver/`, using `guidance/`, `solver_examples/`,
prior score cards, and evaluator reports as active inputs. Carry forward credited strengths,
and focus edits on evaluator reports, unresolved issues, weak steps, and missing cases. 
If the full theorem remains out of reach, still close as many named gaps as possible, 
or build up a structure that could potentially resolve the gaps. 
The solver will be evaluated based on `scoring_rubrics.md`. For all dimensions, higher scores are better.

2. You should also improve the files in `guidance/`, following the instructions above.

**MANDATORY:** Before you stop, you MUST invoke the `reference-validator` and then `check-runner` sub-agents from `.claude/agents/` or `.codex/agents/`, depending on which runtime you are using. Have them execute their configured workflow from the workspace root. These subagents will provide necessary checks without formal evaluation. Do NOT finish without running these checks. If anything fails, repair the output and rerun the check until they pass.

Do not ask the human for clarification, approval, or feedback at any point during this run. Do the work autonomously, finish your edits, provide your final summary, and stop.

## Reference Solver Examples

Before editing, read the reference solver examples together with their
`logs/evaluate/` outputs. Do not rely on `score.yaml` alone; use evaluation logs
to identify rewarded proof moves, penalized gaps, and reusable repair strategies.

Also read the sampled optimizer examples, especially their `guidance/` files, to
identify reusable strategies and guidance improvements.

You have a few reference example(s). Their score cards are shown below:

{solver_examples_block}
{optimizer_examples_block}
