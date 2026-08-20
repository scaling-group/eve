---
name: configure-eve-driver
description: "Use when choosing, editing, or overriding EvE driver settings for Codex, OpenCode, or interactive/debug runs."
---

# Configure the EvE Driver

Use the public presets first:

- `driver: codex_smoke` is for fast runtime validation.
- `driver: codex_max` is for full non-interactive runs.
- `driver: opencode_smoke` is the equivalent fast OpenCode path.
- `driver: opencode_max` is the equivalent full-run OpenCode path.

Do not add a new preset for a one-off variation. Prefer Hydra overrides until the role is stable enough to document as a supported entrypoint.

## Public Surface

Driver presets live in `configs/eve/driver/`. Keep that directory small:

- `codex_smoke.yaml`: non-interactive Codex backend, small turn budget, web search disabled.
- `codex_max.yaml`: non-interactive Codex backend, full-run budget, web search enabled.
- `opencode_smoke.yaml`: non-interactive OpenCode backend with a small native step cap.
- `opencode_max.yaml`: non-interactive OpenCode backend with a full-run native step cap.

Task smoke configs live beside task configs, but keep the root config surface compact:

- `circle_packing.smoke` is the single tracked circle packing smoke preset.
- By default it validates the ordinary loop.
- Use explicit Hydra overrides from that same entrypoint for judge evaluation or role-specific system prompt smoke paths.

Task smoke configs use `driver: codex_smoke` by default; select `driver=opencode_smoke` to exercise OpenCode. Do not shrink `loop.n_workers_phase2` in the default smoke path; the two-worker default covers Phase 2 multi-worker behavior and optimizer sync. Use `loop.n_parallel_phase2` when you only need to reduce concurrent solver agents.

## Normal Commands

Fast smoke:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=circle_packing.smoke
```

OpenCode smoke (use a native `provider/model` identifier):

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=circle_packing.smoke driver=opencode_smoke driver.model=deepseek/deepseek-chat
```

Judge evaluation smoke:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner \
  --config-name=circle_packing.smoke \
  evaluation=circle_packing.judge \
  label=circle-packing-judge-smoke
```

Role-specific system prompt smoke:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner \
  --config-name=circle_packing.smoke \
  evaluation=circle_packing.judge \
  loop.max_iterations=1 \
  label=circle-packing-codex-prompt-smoke \
  +driver.overrides.solver.system_prompt_file=configs/eve/optimizer/circle_packing/prompt/CODEX_SYSTEM_PROMPT.md
```

Full run:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=circle_packing
```

OpenCode full run:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=circle_packing driver=opencode_max driver.model=deepseek/deepseek-chat
```

One-off interactive/debug run:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner \
  --config-name=circle_packing.smoke \
  driver.driver=codex_tmux \
  driver.open_iterm2=true
```

If overriding from a full config, keep the backend switch explicit:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner \
  --config-name=circle_packing \
  driver.driver=codex_tmux \
  driver.open_iterm2=true
```

Do not preserve interactive/debug overrides as new YAML presets unless the team decides they are a supported public entry.

## Selection Semantics

The driver config lives under the top-level `driver:` key.

- `driver.driver` selects the backend and wins over `driver.provider`.
- `driver.provider` is accepted for older configs. Avoid it for new public presets.
- If a config inherits from `codex_max`, changing only `driver.provider` will not switch the backend because `driver.driver=codex_exec` is still set.
- Role overrides are shallow merges: EvE starts from the base `driver` mapping, then overlays `driver.overrides.solver` or `driver.overrides.eval`.
- Do not configure `driver.overrides.optimizer`; the independent optimizer driver role was removed.

Supported backend names are `codex_exec`, `codex_tmux`, and `opencode`.

## Authentication

- Codex uses its native authentication by default; alternate providers may use `model_providers.*.env_key`.
- OpenCode uses credentials saved by `opencode auth login` unless an explicit `model_providers.*.env_key` is configured.
- Explicit environment credentials take precedence. Do not rely on undeclared ambient variables.

## Driver Options

Common fields:

- `driver.executable`: command name to launch; defaults to `codex` or `opencode` according to the backend.
- `driver.model`: backend-native model identifier.
- `driver.rollout_max_turns`: per-rollout agent turn/step cap.
- `driver.timeout_seconds`: per-rollout wall-clock timeout.
- `driver.system_prompt_file`: repo-relative path to a backend system prompt file.
- `driver.model_providers`: provider config map; if an entry declares `env_key`, EvE fails fast when that environment variable is missing.
- `driver.token_pricing`: per-driver pricing metadata override. Prefer shared pricing in `configs/pricing.yaml` unless a run needs a local override.

Codex-only fields (`codex_exec`, `codex_tmux`):

- `driver.reasoning_effort`: reasoning budget where supported.
- `driver.effort_level`: accepted as a fallback spelling for Codex-style reasoning effort.
- `driver.web_search`: `disabled`, `cached`, or `live`.
- `driver.budget_prompt`: whether EvE injects the budget prompt wrapper.
- `driver.enable_multi_agent`: optional backend feature override.
- `driver.personality`: optional backend personality override.
- `driver.model_provider`: optional alternate provider id for Codex-style backends.

OpenCode-only semantics (`opencode`):

- `driver.model` must use OpenCode's native `provider/model` form when set; omit it to use OpenCode's configured default.
- `driver.variant` passes the provider-specific native variant unchanged.
- `driver.rollout_max_turns` is applied through OpenCode's native `agent.build.steps` setting.
- `driver.budget_prompt` must be `false`; Codex hook injection is not translated.
- Codex-only options are rejected instead of ignored or mapped.
- HOME, XDG state, project discovery, and external skills are isolated per EvE workspace.
- EvE-owned `AGENTS.md`, guidance skills, and helper agents are projected into the isolated
  OpenCode configuration.
- Only `model_providers.*.env_key` is reused; other Codex provider settings are rejected
  instead of translated.
- Spawn creates a native OpenCode session; resume uses its exact session ID in the same workspace. Native fork, cross-workspace migration, and snapshots are unsupported.

Tmux-only fields (`codex_tmux`):

- `driver.approval_policy`: approval policy passed to the interactive backend.
- `driver.sandbox_mode`: sandbox mode passed to the interactive backend.
- `driver.allow_network`: when `sandbox_mode` is omitted, `true` maps to `danger-full-access`; otherwise the default is `workspace-write`.
- `driver.open_iterm2`: whether EvE opens the tmux session in a visible terminal window.
- `driver.completion_filename`: internal tmux completion marker filename.
- `driver.instruction_filename`: internal tmux instruction filename.

## Implementation References

This skill is grounded in the current runtime surfaces:

- `src/scaling_evolve/algorithms/eve/runtime/driver.py` for backend selection, role overrides, tmux pool creation, sandbox/web-search normalization, provider environment resolution, and pricing config loading.
- `src/scaling_evolve/providers/agent/drivers/codex_exec.py` for non-interactive Codex fields.
- `src/scaling_evolve/providers/agent/drivers/opencode.py` for non-interactive OpenCode spawn/resume, JSONL parsing, and native step limits.
- `src/scaling_evolve/providers/agent/drivers/codex_tmux.py` for interactive Codex/tmux fields.
- `src/scaling_evolve/providers/agent/config.py` for persistent session provider config validation.
