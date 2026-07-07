---
name: writing-subagents
description: "Use when creating new subagents, editing existing subagents, or verifying subagents work before deployment"
---

# Writing Subagents

## Directory Structure

Put subagent files in `guidance/agents/codex/`. We have linked `.codex/agents/` to this folder, so they can be loaded automatically.

```
guidance/
  agents/
    codex/
      subagent-name.toml
```

## Subagents File Format

Write TOML files with at least:

```toml
name = "subagent-name"
description = "Use when [specific triggering conditions and expected output]"
developer_instructions = '''
# Subagent title

Task-specific instructions here.
'''
```

Reference docs if deeper format details are needed:

- Codex subagents: `https://developers.openai.com/codex/subagents`
