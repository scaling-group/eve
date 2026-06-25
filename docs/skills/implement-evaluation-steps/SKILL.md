---
name: implement-evaluation-steps
description: "Use when defining or updating Eve evaluation steps."
---


# Implement Evaluation Steps

## Overview

Evaluation is a first-class config tree: `configs/eve/evaluation/<app>.yaml` composed via the
run config's `defaults:` list (e.g. `- evaluation: circle_packing`). The file
declares an ordered `evaluation.steps` pipeline plus an `evaluation.failure_score`. Every run
config MUST compose an `evaluation:` group; there is no implicit fallback.

`evaluation.steps` is an ordered, arbitrarily-long pipeline. Dispatch is by FORM, not a `kind`
field:
- a bare `.sh` path string -> programmatic (deterministic) step
- a `{prompt, immutable?}` mapping -> LLM judge step (a first-class worker: its own prompt +
  budget, plus an OPTIONAL immutable rubric + score-check subagent). Omit `immutable` for a
  lightweight judge that runs from its ENTRYPOINT alone.

## Config form

```yaml
# @package _global_
evaluation:
  steps:
    - configs/eve/evaluation/<app>/evaluation.sh        # programmatic step
    # - { prompt: configs/eve/evaluation/<app>/prompt_assess,
    #     immutable: configs/eve/evaluation/<app>/immutable_assess }   # judge step
  failure_score:
    score: 0.0
    summary: evaluation failed
  seed_solver_score: null
  seed_solver_skip_evaluation: false
```

## Runtime Inputs (shell steps)

Shell evaluators receive these environment variables:
- `EVE_WORKSPACE_ROOT`: workspace root
- `EVE_SOLVER_ROOT`: solver candidate repo root, that is `EVE_WORKSPACE_ROOT/solver`
- `EVE_EVAL_LOG_ROOT`: formal evaluation log root, that is `EVE_WORKSPACE_ROOT/logs/evaluate`

Use these paths directly in the shell file instead of reconstructing ad hoc paths.

## Requirements

- All shell steps run from `EVE_WORKSPACE_ROOT`, not `EVE_SOLVER_ROOT` or the downstream repo.
- Write final evaluation files to `EVE_EVAL_LOG_ROOT`. This is the only interface that will be extracted.
- `EVE_EVAL_LOG_ROOT/score.yaml` is required.

`score.yaml` rules:
- `score.yaml` must always exist.
- The score payload may be any YAML PyTree.
- If evaluation fails, write a minimal YAML payload such as:

```yaml
score: 0.0
summary: evaluation failed
```

You can store files of any format within the `EVE_EVAL_LOG_ROOT` directory, including multimedia
like images and videos. If your logging structure is complex, it is recommended to include a
`README.md` inside `EVE_EVAL_LOG_ROOT` to document the file organization and contents.

## Reference Example

circle_packing ships both forms:
- Default single-step: `configs/eve/evaluation/circle_packing.yaml`
  pointing at `configs/eve/evaluation/circle_packing/evaluation.sh`.
- Judge pipeline: `configs/eve/evaluation/circle_packing.judge.yaml`
  (`performance.sh` -> `assess` -> `aggregate`), with prompt/immutable dirs under
  `configs/eve/evaluation/circle_packing/`.
