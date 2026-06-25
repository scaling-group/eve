---
name: writing-skills
description: "Use when creating new skills, editing existing skills, or verifying skills work before deployment"
---

# Writing Skills

## Directory Structure

Put skills in `docs/skills`, we have linked `.agents/skills` and `.claude/skills` to this folder, so they can be loaded automatically.

```
docs/
  skills/
    skill-name/
      SKILL.md              # Main reference (required)
      supporting-file.*     # Only if needed
```

## SKILL.md Structure

```markdown
---
name: skill-name-with-hyphens
description: "Use when [specific triggering conditions and symptoms]"
---

Other details here

```
Don't forget the quotes around `description`.
