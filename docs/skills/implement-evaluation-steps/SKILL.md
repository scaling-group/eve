---
name: implement-evaluation-steps
description: "Use when defining or updating Eve evaluation steps."
---


# Implement Evaluation Step

## Overview

`evaluation_steps` specifies the steps of evaluation.

Supported step types:
- `.sh`, which will be executed
- `.md`, which will spawn an agent with instructions

## Runtime Inputs

Shell evaluators receive these environment variables:
- `EVE_WORKSPACE_ROOT`: workspace root
- `EVE_OUTPUT_ROOT`: output candidate repo root, that is `EVE_WORKSPACE_ROOT/output`
- `EVE_EVAL_LOG_ROOT`: formal evaluation log root, that is `EVE_WORKSPACE_ROOT/logs/evaluate`

Use these paths directly in the shell file instead of reconstructing ad hoc paths.

## Requirements

- All evaluation steps run from `EVE_WORKSPACE_ROOT`, not `EVE_OUTPUT_ROOT` or the downstream repo.
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

You can store files of any format within the `EVE_EVAL_LOG_ROOT` directory, including multimedia like images and videos. If your logging structure is complex, it is recommended to include a `README.md` inside `EVE_EVAL_LOG_ROOT` to document the file organization and contents.

## Reference Example

Use the following sh file as an example:
- `configs/eve/application/circle_packing/evaluation.sh`
