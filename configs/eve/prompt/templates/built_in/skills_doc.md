## Skills Convention

If you want to provide reusable skills, use this layout:

```text
output/
└── skills/
    └── <skill-name>/
        └── SKILL.md
```

Rules:

- Create one subdirectory per skill under `output/skills/`.
- Put the skill instructions in a file named `SKILL.md` inside that subdirectory.
- Keep skill names short, lowercase, and filesystem-safe.
- Use Markdown in `SKILL.md` and write concrete instructions for the agent.
- Treat each skill directory as a self-contained unit; if a skill needs extra
  reference files, place them next to `SKILL.md` inside the same skill directory.

Recommended `SKILL.md` shape:

```md
---
name: <skill-name>
description: "<one-line summary>"
---

details...
```

- The frontmatter is optional but recommended.
- If frontmatter is present, keep it short and descriptive.
- Quote `description` values if they contain punctuation such as `:` so YAML stays valid.
- The body should explain when to use the skill, what to read first, what
  concrete steps to take, and what outcome is expected.
