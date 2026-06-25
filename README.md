<div align="center">

<img src=".github/assets/eve-banner.png" width="600">

# EvE: Evolutionary Ensemble of Agents

### **A decentralized ensemble of coding agents co-evolving with code repositories.**

[![python](https://img.shields.io/badge/-Python_%3E%3D3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![ruff](https://img.shields.io/badge/Code%20Style-Ruff-orange.svg?labelColor=gray)](https://docs.astral.sh/ruff/)<br>
[![arXiv](https://img.shields.io/badge/arXiv-2605.09018-b31b1b.svg)](https://arxiv.org/abs/2605.09018)
[![Slack](https://img.shields.io/badge/Slack-Community-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/eve-mf57726/shared_invite/zt-3xym0tp2c-IZOp3oHMh5Fp7xkQwkGlKg)
[![license](https://img.shields.io/badge/License-Apache_2.0-blue.svg?labelColor=gray)](LICENSE)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/scaling-group/eve/pulls)
[![GitHub stars](https://img.shields.io/github/stars/scaling-group/eve?style=social)](https://github.com/scaling-group/eve/stargazers)

[Scientific Computing and Intelligence Group (Scaling Group) @ NUS](https://scaling-group.github.io)

</div>

<p align="center">
<a href="https://arxiv.org/abs/2605.09018">Paper</a> &middot; <a href="#overview">Overview</a> &middot; <a href="#agent-backend">Agent Backend</a> &middot; <a href="#how-it-works">How It Works</a> &middot; <a href="#quick-start">Quick Start</a> &middot; <a href="#set-up-your-own-task">Set Up Your Own Task</a> &middot; <a href="#example-icon-context-length-generalization">Example</a> &middot; <a href="#codex-operator-skills">Skills</a> &middot; <a href="#papers-using-eve">Papers</a> &middot; <a href="#citation">Citation</a>
</p>

<p align="center"><b>Unlimited agents, fully autonomous.</b></p>

<p align="center">
  <img src=".github/assets/eve-demo.gif" width="720">
</p>
<p align="center">
  <sub>In interactive runs, each tmux pane is an independent coding agent editing, evaluating, and evolving solutions.</sub>
</p>

## Overview

Modern coding agents already have autonomous planning, complex reasoning,
sophisticated context management, and sub-agent invocation. Rather than
reinventing the wheel with "LLMs as optimizers", **EvE wraps existing, highly
capable coding agents into a decentralized evolutionary ensemble** that
co-evolves two populations: a **solver population** of functional components
within a code repository, and an **agent population** whose guidance and skills
are continuously refined through pairwise competition.

Any coding agent or multi-agent system can be seamlessly encapsulated as an
individual within the ensemble. This naturally supports **recursive nesting**:
an entire EvE ensemble can function as a single individual inside a higher-level
ensemble.

<p align="center">
  <img src=".github/assets/eve-overview.png" width="720">
</p>
<p align="center">
  <sub><b>(a)</b> LLM as Optimizer: an LLM proposes a code block, which is scored and re-prompted. <b>(b)</b> Coding Agent: operates on a full code repository with autonomous planning, tool use, and sub-agent invocation. <b>(c)</b> Evolutionary Ensemble (this work): a decentralized ensemble of coding agents that evolves with another population of functional components within a code repository.</sub>
</p>

## Agent Backend

The current public EvE release is Codex-first. The tracked driver presets use
[**Codex**](https://github.com/openai/codex) as the agent backend:

- `codex_smoke` for fast runtime validation.
- `codex_max` for full non-interactive runs.

Codex can be run in two modes:

- **Interactive (tmux)**: each agent session runs in a visible tmux pane,
  exactly like using Codex in your terminal. You can watch agents work,
  intervene, and debug in real time.
- **Non-interactive (subprocess)**: agents run as headless subprocesses with
  JSON streaming. This is the recommended default for unattended runs and CI.

> [!IMPORTANT]
> EvE orchestrates third-party coding agents; it does **not** provide unlimited
> access to any AI service. Each agent session consumes subscription quota or
> API credits on **your own account**. EvE does not bypass or modify any
> provider's authentication, rate limits, or usage restrictions.

## First-time Setup

1. **Install dependencies.**

       uv sync

2. **Agent authentication.** Install and authenticate your coding agent. For
   example, run `codex login` if you use Codex through a ChatGPT subscription,
   or export the API credentials required by your selected backend.

3. **Hook trust (for Codex >= 0.130.0).** EvE uses hooks for workspace
   sandboxing and budget prompts. Run once per machine from the repository root
   if you are using Codex:

       uv run python -m scaling_evolve.providers.agent.codex_hooks
       codex .
       # Type /hooks -> press t to trust all -> Esc -> Ctrl-C

4. **Verify.** Run a short smoke test using Codex to confirm everything works:

       uv run python -m scaling_evolve.algorithms.eve.runner \
         --config-name=circle_packing.smoke

   This runs a short circle packing demo with headless Codex agents.

## How It Works

<p align="center">
  <img src=".github/assets/eve-framework.png" width="720">
</p>

EvE maintains two co-evolving populations: a **solver population** containing
functional components in a code repository, and an **agent population** where
each agent carries cumulative working logs and an Elo-based score. EvE fixes the
base agent substrate and focuses on evolving the cumulative guidance and skills
that dictate agent behaviors.

Each agent operates in a dedicated workspace with all dependencies included, and
its modification scope is explicitly restricted to designated files and enforced
by post-generation checks. Solver improvement and self-referential agent
optimization happen in one unified step: agents improve guidance and skills
while editing code repositories, and this guidance is then repeatedly evaluated
during future iterations, with concrete scores that drive sampling probability.

The formal procedure is given in the algorithm below.

<details>
<summary><b>Algorithm: Evolutionary Ensemble of Agents</b> (click to expand)</summary>
<br>
<p align="center">
  <img src=".github/assets/eve-algorithm.png" width="560">
</p>

In each iteration, EvE samples a set of high-performing working agents, along
with reference sets of solvers and agents, which are combined with the base code
repository to provide context. A synchronous race is then conducted: each
working agent operates within its own workspace on the same reference set,
producing a new solver candidate and a potentially revised agent. By forcing all
agents to refine the same references, variance in solver quality is directly
attributed to the effectiveness of each agent's strategy. After evaluation, a
pairwise win-loss matrix is constructed and agent Elo ratings are updated.
Agents that revised their guidance are integrated back into the population,
preserving new strategies and their procedural evidence.

</details>

### Workspace Layout

Each agent runs in its own workspace. A solver workspace looks like:

```text
workspace/
├── solver/             # the solver candidate to edit; extracted as the submission
├── guidance/           # optimizer guidance to apply and refine
├── solver_examples/    # sampled reference solvers from the population (read-only)
│   └── <solver_id>/
│       ├── solver/         # that solver's files
│       ├── logs/           # its evaluation logs
│       └── score.yaml      # its score
├── guidance_examples/  # sampled reference optimizer guidance (read-only)
│   └── <optimizer_id>/
│       ├── guidance/       # that optimizer's guidance files
│       ├── logs/           # its logs
│       └── score.yaml      # its score
├── logs/               # this run's optimization and evaluation logs
└── README.md           # workspace-specific instructions (read first)
```

The `solver_examples/` and `guidance_examples/` directories are populated from
the active loop config before each worker run.

## Quick Start

### Check Your Setup

Use the built-in circle packing smoke to verify that your local setup is wired
correctly: dependencies, config composition, hooks/authentication, and the Codex
backend all run end to end.

```bash
uv sync

uv run python -m scaling_evolve.algorithms.eve.runner --config-name=circle_packing.smoke
```

This is a local setup check, not a quality benchmark.

### Math Proof quickstart

To try EvE on a mathematical problem, start Codex from the repository root and
ask it to use the Math Proof quickstart:

```bash
codex .
```

```text
Run the Math Proof quickstart for this problem:

<paste the problem statement here>
```

The quickstart is a Codex operator workflow: Codex creates a local task copy
under `examples/tmp/`, uses `configs/eve/math_proof_quickstart.yaml` with
`application.path`, launches a real EvE attempt, and then supervises and
inspects the run by following the bundled `run-math-proof-quickstart` skill.

## Set Up Your Own Task

To run EvE on your own codebase, you need a source repository, an application
config, an evaluation config, initial guidance, and a top-level experiment
config.

### 1. Application config

Point EvE at your code repository and declare which files agents may edit:

```yaml
# configs/eve/application/your_task.yaml
application:
  name: your-task
  github_url: https://github.com/your-org/your-repo
  commit: abc123... # pin to a specific commit
  editable:
    files:
      - src/model.py
      - configs/params.yaml
    folders: []
  boundary_failure_score:
    score: 0.0
    summary: boundary check failed
```

Agents are strictly confined to `editable.files` and `editable.folders`; any
modification outside this surface is rejected by the boundary checker.
For a local task tree instead of a Git checkout, set `application.path` and omit
`github_url` and `commit`; EvE accepts either `path` or the pair
`github_url`/`commit`, but not both.

### 2. Evaluation config

Provide one or more evaluation steps. A shell step runs inside the candidate
workspace and must write a `score.yaml` file with at least `score: <float>` and
`summary: <string>` to the evaluation log directory.
Evaluation steps can also be judge-agent steps written as `{prompt,
immutable?}` mappings; see
[`implement-evaluation-steps`](docs/skills/implement-evaluation-steps/) for the
full evaluation-step surface.

```yaml
# configs/eve/evaluation/your_task.yaml
evaluation:
  steps:
    - configs/eve/evaluation/your_task/evaluation.sh
  failure_score:
    score: 0.0
    summary: evaluation failed
  seed_solver_score: null
  seed_solver_skip_evaluation: false
```

### 3. Initial guidance

Write Markdown documents and optional skills that describe the task and search
directions for the agents. Place them in an `initial_guidance/` directory:

```text
configs/eve/optimizer/your_task/initial_guidance/
  docs/                          # task context, search directions, background knowledge
  skills/
    your-skill/SKILL.md          # reusable instructions agents will read and evolve
```

Configure the initial guidance, immutable prompt assets, and worker prompts:

```yaml
# configs/eve/optimizer/your_task.yaml
optimizer:
  initial_guidance: configs/eve/optimizer/your_task/initial_guidance
  workers:
    selection: random
    items:
      - name: normal
        weight: 1.0
        immutable: configs/eve/optimizer/your_task/immutable
        prompt: configs/eve/optimizer/your_task/prompt
  immutable_renderer:
    _target_: scaling_evolve.algorithms.eve.workspace.immutable_renderers.default.DefaultRenderer
  evaluation:
    _target_: scaling_evolve.algorithms.eve.populations.evaluators.elo.ScalarEloEvaluator
    k_factor: 32.0
    initial_score:
      elo: 1500.0
```

Files under `initial_guidance/` seed the initial agent population. As agents
discover what works and what does not, they revise the guidance for future
iterations.

### 4. Loop config

The default loop controls how many workers run per iteration and how many
population examples each worker sees:

```yaml
# configs/eve/loop/default.yaml
loop:
  max_iterations: 25
  n_workers_phase2: 2
  n_solver_examples_phase2: 4
  n_optimizer_examples_phase2: 4
  produce_optimizer_in_phase2: ${loop.n_workers_phase2}
```

At each iteration, EvE launches `n_workers_phase2` solver workers in parallel.
Each worker gets sampled solver references under `solver_examples/` and sampled
guidance references under `guidance_examples/`. Workers race on the same
iteration without communicating directly; the resulting solvers are evaluated,
and produced guidance candidates are kept according to
`produce_optimizer_in_phase2`.

### 5. Experiment config

Compose everything with Hydra and launch:

```yaml
# configs/eve/your_task.yaml
defaults:
  - runtime: default
  - loop: default
  - driver: codex_max
  - logger: many_loggers
  - application: your_task
  - optimizer: your_task
  - evaluation: your_task
  - _self_

label: your-task
```

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=your_task
```

See `configs/eve/circle_packing.yaml`, `configs/eve/icon.yaml`, and
`configs/eve/math_proof_quickstart.yaml` for complete working examples.

## Codex Operator Skills

The [`docs/skills/`](docs/skills/) directory is the operator layer for working
with EvE through Codex. Start Codex at the repository root, discuss the task or
experiment you want to run, and ask it to use the relevant skill. The skills
cover launching, supervising, resuming, importing, inspecting, debugging, and
authoring EvE tasks; the Math Proof quickstart above is one example of this
workflow.

| Skill | Description |
|-------|-------------|
| [`configure-eve-driver`](docs/skills/configure-eve-driver/) | Configure driver presets and one-off interactive/debug overrides |
| [`supervise-run`](docs/skills/supervise-run/) | Watch a live run and recover or escalate stalls |
| [`resume-run`](docs/skills/resume-run/) | Continue the same run after a pause or interruption |
| [`import-run`](docs/skills/import-run/) | Start a new run seeded from prior run populations |
| [`inspect-population`](docs/skills/inspect-population/) | Inspect evolved solver and guidance populations |
| [`debug-agent-session`](docs/skills/debug-agent-session/) | Debug a single agent rollout from its transcript |
| [`implement-evaluation-steps`](docs/skills/implement-evaluation-steps/) | Define or update evaluation steps |
| [`implement-check-subagent`](docs/skills/implement-check-subagent/) | Implement a sanity check subagent for the optimization phase |
| [`implement-subagent`](docs/skills/implement-subagent/) | Implement a custom Codex subagent |
| [`run-circle-packing-smoke`](docs/skills/run-circle-packing-smoke/) | Run a quick end-to-end smoke test |
| [`run-math-proof-quickstart`](docs/skills/run-math-proof-quickstart/) | Run the Math Proof quickstart |
| [`theorem-search`](docs/skills/theorem-search/) | Find and verify mathematical theorem statements and proof dependencies |
| [`matlas-search`](docs/skills/matlas-search/) | Search published mathematical literature for theorem statements and citation metadata |
| [`writing-skills`](docs/skills/writing-skills/) | Create or edit skills |

These skills are also auto-loaded by supported agent entrypoints when working
inside the repository.

## Community

Questions, feedback, or running into issues? Join our
[Slack workspace](https://join.slack.com/t/eve-mf57726/shared_invite/zt-3xym0tp2c-IZOp3oHMh5Fp7xkQwkGlKg).

## Example: ICON Context Length Generalization

Applied to [ICON](https://github.com/scaling-group/icon-core) (In-Context
Operator Networks), EvE autonomously discovered a novel positional-encoding
mechanism that reduced generalization error by over 80% compared to the
hand-designed baseline, turning a catastrophic out-of-distribution failure into
robust performance. This application uses EvE's external-repo mode with remote
GPU evaluation. See `examples/icon/README.md` for reproduction instructions.

We compare three experimental conditions, each run twice independently under
identical compute budgets:

- **EvE**: the full ensemble with continuous agent evolution.
- **Static-Initial**: the initial agent is used throughout the entire search,
  with no agent evolution.
- **Static-Final**: the single best-rated agent from the corresponding
  completed EvE run is extracted and frozen for a fresh search.

<p align="center">
  <img src=".github/assets/fig-convergence.png" width="560">
</p>
<p align="center">
  <sub>Search trajectories for all variants (two independent runs each). The y-axis is the running minimum of mean error (lower is better); the x-axis is cumulative equivalent tokens in millions. The gray dashed line marks the Seed baseline.</sub>
</p>

The two EvE runs descend in near-lockstep, converging to almost identical final
errors. The Static-Initial runs diverge: one eventually approaches EvE while the
other plateaus at a higher level. Static-Final, despite starting from a
higher-rated agent, suffers from phase mismatch: the frozen agent was optimized
for the late stage of the original EvE run but a fresh search requires
early-stage exploration strategies that this agent no longer carries. Continuous
evolution is indispensable for both performance and robustness.

The complete raw search traces for all six runs (every solver's source code,
agent conversations, guidance updates, and evaluation scores) are available in
the [v0.1.0 release](https://github.com/scaling-group/eve/releases/tag/v0.1.0).

## Papers Using EvE

EvE has been applied to a growing set of scientific problems. See
[`docs/papers.md`](docs/papers.md) for examples, paper links, and BibTeX entries.

## Citation

Please cite our paper if you use EvE in your research:

```bibtex
@article{yu2026eve,
  title         = {Evolutionary Ensemble of Agents},
  author        = {Yu, Zongmin and Yang, Liu},
  year          = {2026},
  url           = {https://arxiv.org/abs/2605.09018},
  eprint        = {2605.09018},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NE}
}
```

## Acknowledgement

This project is part of the
[Scientific Computing and Intelligence Group (Scaling Group)](https://scaling-group.github.io)
at the National University of Singapore.

Please include the NOTICE file (already included in this repository) in your
code base that uses this repository.
