---
name: read-eval
description: "Phase 2 solver optimization only: read the evaluator first, infer what it rewards, then make targeted solver edits."
---

Use this skill only for Phase 2 solver optimization.
Do not use it for Phase 4 optimizer optimization.

1. Read `output/evaluate.py` carefully before proposing edits.
2. Infer what the evaluator is rewarding, penalizing, or constraining.
3. Identify the smallest high-leverage changes to `output/candidate.py`.
4. Prefer concrete task-specific improvements over process commentary.
5. Keep edits local and score-oriented.
