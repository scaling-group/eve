---
name: writing-subagents
description: "Use when creating new subagents, editing existing subagents, or verifying subagents work before deployment"
---

# Writing Subagents

## Directory Structure

Put subagent files in `guidance/agents/claude/` or `guidance/agents/codex/` (depending on what agent you are). We have linked `.claude/agents/` and `.codex/agents/` to these folders, so they can be loaded automatically.

```
guidance/
  agents/
    claude/
      subagent-name.md      # if you are Claude agent
    codex/
      subagent-name.toml    # if you are Codex agent
```

## Subagents File Format

For Codex subagents, write TOML files with at least:

```toml
name = "subagent-name"
description = "Use when [specific triggering conditions and expected output]"
developer_instructions = '''
# Subagent title

Task-specific instructions here.
'''
```

For Claude Code subagents, write Markdown files with frontmatter:

```markdown
---
name: subagent-name
description: "Use when [specific triggering conditions and expected output]"
---

# Subagent title

Task-specific instructions here.
```

Reference docs if deeper format details are needed:

- Codex subagents: `https://developers.openai.com/codex/subagents`
- Claude Code subagents: `https://code.claude.com/docs/en/sub-agents`
