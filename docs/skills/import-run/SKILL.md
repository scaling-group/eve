---
name: import-run
description: "Use when starting a new Eve run seeded from one or more prior runs, e.g. keep the good solvers but try a fresh optimizer"
---

# Import into a new run

`import` starts a **new** run (new run id) whose initial population is seeded from one or more prior runs. Use it to keep what worked and change what didn't. To continue the same run instead, see `../resume-run/SKILL.md`.

The classic case: the optimizer evolved poorly but some solvers are good. Keep the solvers, start a fresh optimizer.

## Command

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=<config> \
  'import_from={path: <old_run_root>, import_solvers: true, import_optimizers: false}' \
  +optimizer.seed_initial_on_import=true
```

- `import_solvers` / `import_optimizers` select, per side, which populations carry over (both default `true`); they live inside the per-source mapping.
- **Gotcha (do not skip):** with `import_optimizers: false`, the optimizer population starts **empty** unless you also pass `+optimizer.seed_initial_on_import=true`, which seeds the new run's `initial_guidance/`. Omit it and the run has no optimizer. The `+` is required (this key is not pre-declared in the config).
- To change the optimizer for the new run, edit `initial_guidance/` before launching. Unlike resume, import **does** seed it.

## Multiple sources

`import_from` accepts a list; sources import in order, additively:

```bash
'import_from=[{path: <run_a>, import_solvers: true, import_optimizers: false}, {path: <run_b>, import_solvers: true, import_optimizers: false}]'
```

- Each source DB must contain exactly one run id (a guard; a normal single run satisfies it).
- Gotcha: if two sources contain entries with the same id, the later one silently overwrites the earlier. Only compose sources you know are disjoint.

## After launch

It is a normal new run: supervise with `../supervise-run/SKILL.md`, inspect with `../inspect-population/SKILL.md`.

## Note

- `import_from` and `resume_from` are mutually exclusive.
