# Evolutionary Ensemble (EvE) Overview

## Core Concepts

EvE maintains two coupled populations:

- `solver`: code or files that directly work on the downstream task repo. In
  Phase 2, the agent edits a solver candidate inside `output/` and that
  candidate is then evaluated on the concrete task.
- `optimizer`: markdown guidance files that improve solvers, including
  description, skills, suggestions, etc. They will be applied into a
  Phase 2 workspace to help the solver-optimization agent.

The loop runs in four phases:

1. Phase 1 samples reference optimizers and reference solvers from the current
   populations.
2. Phase 2 applies each sampled optimizer to sampled solver examples to produce
   a new solver candidate, then evaluates that candidate on the task.
3. Phase 3 updates optimizer scores from the relative performance of the solver
   candidates produced in Phase 2.
4. Phase 4 improves the optimizer population itself by editing optimizer
   guidance that will later guide future Phase 2 solver optimization.
   The main guidance context for this Phase 4 run is provided by the
   lead optimizer, or empty, or fixed.

## Workspace

### Terminology

- `run root`: the outer Eve run directory for one whole experiment. It contains run-level artifacts such as databases, artifact stores, and all per-phase workspaces. Agents do not work directly in the run root.
- `phase workspace root`: the directory for one concrete Phase 2 or Phase 4 agent run. When these instructions say `guidance/`, the reference example directories, `output/`, `logs/`, `README.md`, or `score.yaml`, they mean paths relative to the current phase workspace root.
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

### Optimizer Workspace (Phase 4)

```text
phase_workspace_root/
├── AGENTS.md        ← workspace agent instructions
├── CLAUDE.md        ← workspace agent instructions
├── README.md        ← workspace-specific notes, must read.
├── guidance/        ← optimizer guidance: from lead optimizer, or empty, or fixed
│   └── skills/      ← optional skill tree; exposed through `.claude/skills`
│                       and `.codex/skills`
├── .claude/
│   └── skills -> ../guidance/skills
├── .codex/
│   └── skills -> ../guidance/skills
├── task_base/       ← downstream task repo, with seeded solver.
├── examples/        ← sampled reference optimizers
│   └── <optimizer_id>/
│       ├── optimizer/ ← optimizer files
│       ├── logs/     ← accumulated Phase 2 solver history for that optimizer
│       │   └── <solver_log_dir>/             ← one historical solver-optimization run
│       │       ├── solver/                   ← produced solver files
│       │       ├── logs/
│       │       │   ├── optimize/             ← solver optimize logs
│       │       │   └── evaluate/             ← solver evaluate logs
│       │       └── score.yaml                ← produced solver score
│       └── score.yaml ← current optimizer score
├── logs/
│   └── optimize/    ← free-form log tree from this Phase 4 optimization run
├── output/          ← optimizer markdown files only; prefilled from a optimizer
│                       in `examples/`. This is the submission tree.
└── score.yaml       ← scores for sampled optimizers, the lead optimizer, the
                        prefill optimizer, and the produced optimizer's initial score
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
