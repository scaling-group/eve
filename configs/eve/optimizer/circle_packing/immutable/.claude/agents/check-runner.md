---
name: check-runner
description: "Run the Phase 2 solver check workflow and report PASS or FAIL."
tools: Bash, Read
---

# Circle Packing Sanity Check

From the workspace root, run this self-check workflow before you stop editing and again
after any meaningful change.
You should run these two commands separately.

1. Run the repository's basic validation:

```bash
python3 -m py_compile solver/candidate.py solver/evaluate.py
```

2. Run the boundary check command:

```bash
{{BOUNDARY_CHECK_COMMAND}}
```
