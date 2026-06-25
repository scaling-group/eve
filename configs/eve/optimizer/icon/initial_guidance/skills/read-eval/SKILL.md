---
name: read-eval
description: "Use when interpreting an ICON PE solver evaluation to update optimizer guidance."
---

Use this skill when a solver eval or historical solver log needs to inform the next optimizer update.

1. Extract `score`, `score_metric`, `score_components`, and `quest_qoi_v_by_demo` from `logs/evaluate/eval_summary.json` or `score.yaml`.
2. Read the live target first: higher score means lower `mean_d1_d10`.
3. Inspect `mean_d1_d4` and the full `d1..d10` curve before declaring a change promising.
4. Note whether the candidate improved the full example-count curve broadly or only at a single example count such as `d10`.
5. When writing guidance, separate visible evidence from speculation. Cite the specific solver example and the specific score component you used.
6. Prefer short, reusable lessons over one-off stories tied to a single lucky run.

Primary references:

- `guidance/docs/problem.md`
- `guidance/docs/directions.md`
- `guidance/docs/mutation_surface.md`
- `guidance/docs/literature_notes.md`
