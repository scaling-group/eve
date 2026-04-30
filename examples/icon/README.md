# ICON Overlay

This directory is a minimal overlay for reproducing the ICON positional-encoding
application used by Eve. It mirrors the layout of the ICON core repository so it
can be copied directly into a fork.

## Contents

| Path                                    | Purpose                                                      |
| --------------------------------------- | ------------------------------------------------------------ |
| `configs/experiment/evolve_base.yaml`   | Training experiment overlay used by Eve-generated candidates |
| `configs/model/icon_evolve.yaml`        | Model overlay that routes ICON through the mutable classes   |
| `src/models/icon/icon_evolve.py`        | No-op ICON subclass reserved for solver edits                |
| `src/models/icon/pe_evolve.py`          | Empty positional-encoding mutation module                    |
| `src/models/base/transformer_evolve.py` | No-op transformer encoder and layer subclasses               |
| `scripts/evolve_iter.sh`                | Reference PBS training job                                   |
| `scripts/eval_context_length.sh`        | Reference PBS context-length evaluation job                  |

## Setup

1. Fork the public ICON core repository (`scaling-group/icon-core`) to
   your own GitHub account or organization.
2. Copy the overlay into your fork:

   ```bash
   cp -R examples/icon/* /path/to/your/icon-core-fork/
   ```

3. Commit and push the fork.
4. In Eve, update `configs/eve/application/icon.yaml`:

   ```yaml
   application:
     github_url: https://github.com/YOUR_USER/your-icon-core-fork
     commit: YOUR_FORK_COMMIT_SHA
   ```

5. Set the remote execution environment expected by the reference scripts:

   ```bash
   export EVE_REMOTE_HOST=your-login-host
   export EVE_REMOTE_PYTHON=/scratch/$USER/envs/venvs/icon-core/bin/python
   export EVE_REMOTE_DATA_DIR=/scratch/$USER/data
   export WANDB_ENTITY=your-wandb-entity
   ```

6. Adapt the PBS directives in `scripts/evolve_iter.sh`,
   `scripts/eval_context_length.sh`, and the Eve-side orchestrator scripts under
   `configs/eve/application/icon/`.

The reference implementation assumes PBS over SSH. SLURM or local-GPU users can
reuse the same overlay files, but must adapt the dispatch scripts while preserving
the evaluator contract documented in `docs/icon-application.md`.

For detailed results and analysis, see the EvE paper.
