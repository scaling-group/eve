---
name: implement-check-subagent
description: "Use when implementing a sanity check subagent for the phase 2 optimization in Eve loop."
---

`check` is a sanity check, not formal evaluation.

Implement at least one backend-specific check subagent. For the subagent file format, refer to the `implement-subagent` skill.

For a concrete application example, refer to:
- Claude: `configs/eve/application/circle_packing/check_claude.md`
- Codex: `configs/eve/application/circle_packing/check_codex.toml`


## What to include in the subagent file

Keep in mind that the check subagent is expected to run from `workspace/solver/`, i.e. the downstream task repository.
The implemented subagent file should in general provide:

1. A sanity check command sequence that runs inside `workspace/solver/`. Follow the application pattern from the examples, for example:
   - syntax validation
   - fast smoke runs

2. A boundary check command.
   You do not need to provide the command, following the `{{BOUNDARY_CHECK_COMMAND}}` pattern used in the examples.
