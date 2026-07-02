<div align="center">

<img src=".github/assets/eve-banner.png" width="600">

# EvE: Evolutionary Ensemble of Agents

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
<a href="https://arxiv.org/abs/2605.09018">Paper</a> &middot; <a href="#overview">Overview</a> &middot; <a href="#quick-start">Quick Start</a> &middot; <a href="#how-it-works">How It Works</a> &middot; <a href="#set-up-your-own-task">Set Up Your Own Task</a> &middot; <a href="#example-icon-context-length-generalization">Example</a> &middot; <a href="#papers-using-eve">Papers</a> &middot; <a href="#community">Community</a> &middot; <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src=".github/assets/eve-demo.gif" width="720">
</p>
<p align="center">
  <sub>An illustrative run: each tmux pane is an independent coding agent editing, evaluating, and evolving solutions.</sub>
</p>

## Overview

EvE wraps existing, highly capable coding agents into a decentralized evolutionary ensemble that co-evolves two populations: a **solver population** of functional components within a repository, and an **agent population** whose guidance and skills are continuously self-refined.

Use EvE for challenging tasks where results can be tested or judged: designing
algorithms, improving code, or solving a mathematical problem.

To run EvE, you provide:

- a working environment, such as a codebase or a mathematical problem in a github repository or a local folder;

- the solution files or folders that are allowed to edit,
  such as code files or a proof draft;

- scoring steps that evaluate each solution, such as shell scripts, agent judge prompts, or a combination of them.

EvE then searches for strong solutions without requiring a task-specific workflow or hand-crafted skills.

## Quick Start

> [!IMPORTANT]
> EvE orchestrates third-party coding agents; it does **not** provide unlimited
> access to any AI service. Each agent session consumes subscription quota or
> API credits on **your own account**. EvE does not bypass or modify any
> provider's authentication, rate limits, or usage restrictions.

### First-time setup

1.  **Install dependencies.**

    uv sync

2.  **Agent authentication.** The current public release uses
    [**Codex**](https://github.com/openai/codex) as the default agent backend.
    Install and authenticate Codex with your own login, subscription, or API
    credentials.

3.  **Hook trust (for Codex >= 0.130.0).** EvE uses hooks for workspace
    sandboxing and budget prompts. Run once per machine from the repository root
    if you are using Codex:

        uv run python -m scaling_evolve.providers.agent.codex_hooks
        codex
        # Type /hooks -> press t to trust all -> Esc -> Ctrl-C

4.  **Verify.** Run a short smoke test using Codex to confirm everything works:

        uv run python -m scaling_evolve.algorithms.eve.runner \
          --config-name=circle_packing.smoke

    This runs a short circle packing demo with headless Codex agents.

### Math Proof quickstart

To try EvE on a mathematical problem, start Codex from the repository root and
ask it to use the Math Proof quickstart:

```bash
codex
```

```text
Run the Math Proof quickstart for this problem:

<paste the problem statement here>
```

You may ask codex for more details.

## How It Works

<p align="center">
  <img src=".github/assets/eve-overview.png" width="720">
</p>
<p align="center">
  <sub><b>(a)</b> One-shot code proposal. <b>(b)</b> A coding agent works inside a repository. <b>(c)</b> EvE runs many coding agents across many candidate solutions, scores the results, and carries the useful history into the next round.</sub>
</p>

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

<summary><b>Algorithm: Evolutionary Ensemble of Agents</b></summary>
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

## Set Up Your Own Task

To run EvE on your own task, you need an application config, an evaluation
config, an optimizer config, and a top-level experiment config. Start with the
default loop if you need to tune the search budget or sampling policy.

### 1. Application config

The application config names the task, points EvE at the snapshot, and declares
which files agents may edit:

```yaml
# configs/eve/application/your_task.yaml
application:
  name: your-task
  path: examples/your_task/seed
  editable:
    files:
      - src/model.py
    folders: []
  boundary_failure_score:
    score: -1.0
    summary: boundary check failed
```

For a Git-backed task, use `github_url` and `commit` instead of `path`:

```yaml
application:
  name: your-task
  github_url: https://github.com/your-org/your-repo
  commit: abc123... # pin to a specific commit
  editable:
    files:
      - src/model.py
    folders: []
```

EvE accepts either `path` or the pair `github_url`/`commit`, but not both.
Any edit outside `editable.files` and `editable.folders` is rejected by the
boundary checker. Set `boundary_failure_score` to a hard-fail value in the same
score schema as normal evaluation results; do not blindly copy the number above
if your task uses a different score direction or range.

### 2. Evaluation config

Provide one or more evaluation steps. A shell step runs inside the candidate
workspace and writes the score file expected by your score schema:

```yaml
# configs/eve/evaluation/your_task.yaml
evaluation:
  steps:
    - configs/eve/evaluation/your_task/evaluation.sh
  failure_score:
    score: -1.0
    summary: evaluation failed
  seed_solver_score: null
  seed_solver_skip_evaluation: false
```

Evaluation steps may also use judge agents by providing `immutable` and
`prompt` directories instead of a shell script. Use that path when the score
needs subjective or structured agent judgment rather than a deterministic
programmatic metric.

### 3. Optimizer config

The optimizer config seeds the guidance population and defines worker variants:

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
    _target_: scaling_evolve.algorithms.eve.populations.evaluators.elo.VectorEloEvaluator
    k_factor: 32.0
    initial_score:
      elo: 1500.0
```

`initial_guidance` seeds the optimizer population and can be revised by EvE over
time. A worker's `immutable` directory is fixed scaffold copied into each worker
workspace; it is not itself evolved. A worker's `prompt` directory contains
prompt text used by the driver, such as entrypoint and boundary-repair prompts.

### 4. Experiment config

Compose the run with Hydra. Use the default loop first; override loop settings
only when you need to change the run budget or sampling behavior.

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

logger:
  wandb:
    enabled: false
```

Launch it from the repository root:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=your_task
```

See `configs/eve/math_proof_quickstart.yaml`,
`configs/eve/math_proof_jensen_covering.yaml`,
`configs/eve/circle_packing.yaml`, and `configs/eve/circle_packing.smoke.yaml`
for complete working examples.

## Example: Model Positional Embedding Design

Applied to [ICON](https://github.com/scaling-group/icon-core) (In-Context
Operator Networks), EvE autonomously discovered a novel positional-encoding
mechanism that reduced generalization error by over 80% compared to the
hand-designed baseline, turning a catastrophic out-of-distribution failure into
robust performance. See `examples/icon/README.md` for reproduction instructions.

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

## Community

Questions, feedback, or running into issues? Join our
[Slack workspace](https://join.slack.com/t/eve-mf57726/shared_invite/zt-3xym0tp2c-IZOp3oHMh5Fp7xkQwkGlKg).

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

## Papers Using EvE

EvE has been applied to a growing set of scientific problems.
See additional papers using EvE and their BibTeX entries in [`docs/papers.md`](docs/papers.md).
If you use EvE in your work, we welcome pull requests to add your paper to the list.

## Acknowledgement

This project is part of the
[Scientific Computing and Intelligence Group (Scaling Group)](https://scaling-group.github.io)
at the National University of Singapore.

Please include the NOTICE file (already included in this repository) in your
code base that uses this repository.
