# ICON Overlay

This directory is a minimal overlay for reproducing the ICON positional-encoding
application used by EvE. It mirrors the layout of the public ICON core
repository, [`scaling-group/icon-core`](https://github.com/scaling-group/icon-core),
so it can be copied directly into a fork.

The overlay is not a standalone EvE run directory. Copy it into an ICON fork,
point `configs/eve/application/icon.yaml` at that fork and commit, then run the
EvE `icon` config. During evaluation, EvE stages candidate edits into the remote
ICON checkout and calls the reference training/evaluation scripts from this
overlay.

## Contents

| Path                                    | Purpose                                                      |
| --------------------------------------- | ------------------------------------------------------------ |
| `configs/experiment/evolve_base.yaml`   | Training experiment overlay used by EvE-generated candidates |
| `configs/model/icon_evolve.yaml`        | Model overlay that routes ICON through the mutable classes   |
| `src/models/icon/icon_evolve.py`        | No-op ICON subclass reserved for solver edits                |
| `src/models/icon/pe_evolve.py`          | Empty positional-encoding mutation module                    |
| `src/models/base/transformer_evolve.py` | No-op transformer encoder and layer subclasses               |
| `scripts/evolve_iter.sh`                | Reference PBS training job                                   |
| `scripts/eval_context_length.sh`        | Reference PBS context-length evaluation job                  |

## Setup

1. Fork the public ICON core repository,
   [`scaling-group/icon-core`](https://github.com/scaling-group/icon-core), to
   your own GitHub account or organization.
2. Copy the overlay into your fork:

   ```bash
   cp -R examples/icon/* /path/to/your/icon-core-fork/
   ```

3. Commit and push the fork.
4. In EvE, update `configs/eve/application/icon.yaml`:

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
   ```

6. Adapt the PBS directives in `scripts/evolve_iter.sh`,
   `scripts/eval_context_length.sh`, the EvE check-runner scripts under
   `configs/eve/optimizer/icon/immutable/check_runner/`, and the EvE
   evaluation scripts under `configs/eve/evaluation/icon/`.

Weights & Biases logging is optional in the overlay. Set `WANDB_ENTITY` only if
your environment should publish training logs; otherwise the reference training
script uses local CSV logging.

The reference implementation assumes PBS over SSH. SLURM or local-GPU users can
reuse the same overlay files, but must adapt the dispatch scripts while preserving
the evaluator contract in `configs/eve/evaluation/icon/evaluation.sh`.

For detailed results and analysis, see the EvE paper.
