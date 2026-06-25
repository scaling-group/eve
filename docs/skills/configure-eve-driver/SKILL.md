---
name: configure-eve-driver
description: "Use when choosing, editing, or overriding EvE driver settings for codex_smoke, codex_max, or interactive/debug runs."
---

# Configure the EvE Driver

Use the public presets first:

- `driver: codex_smoke` is for fast runtime validation.
- `driver: codex_max` is for full non-interactive runs.

Do not add a new preset for a one-off variation. Prefer Hydra overrides until the role is stable enough to document as a supported entrypoint.

## Public Surface

Driver presets live in `configs/eve/driver/`. Keep that directory small:

- `codex_smoke.yaml`: non-interactive Codex backend, small turn budget, web search disabled.
- `codex_max.yaml`: non-interactive Codex backend, full-run budget, web search enabled.

Task smoke configs live beside task configs, but keep the root config surface compact:

- `circle_packing.smoke` is the single tracked circle packing smoke preset.
- By default it validates the ordinary loop.
- Use explicit Hydra overrides from that same entrypoint for judge evaluation or role-specific system prompt smoke paths.

Task smoke configs should use `driver: codex_smoke`. Do not shrink `loop.n_workers_phase2` in the default smoke path; the two-worker default covers Phase 2 multi-worker behavior and optimizer sync.

## Normal Commands

Fast smoke:

```bash
uv run python -m scaling_evolve.algorithms.eve.runner --config-name=circle_packing.smoke
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

Supported backend names in code include `codex_exec`, `codex_tmux`, `claude_code`, and `claude_code_tmux`. The public presets should still remain `codex_smoke` and `codex_max`.

## Driver Options

Common Codex-style fields (`codex_exec`, `codex_tmux`):

- `driver.executable`: command name to launch.
- `driver.model`: model identifier passed to the backend.
- `driver.reasoning_effort`: reasoning budget where supported.
- `driver.effort_level`: accepted as a fallback spelling for Codex-style reasoning effort.
- `driver.rollout_max_turns`: per-rollout agent turn cap.
- `driver.timeout_seconds`: per-rollout wall-clock timeout.
- `driver.web_search`: `disabled`, `cached`, or `live`.
- `driver.budget_prompt`: whether EvE injects the budget prompt wrapper.
- `driver.enable_multi_agent`: optional backend feature override.
- `driver.personality`: optional backend personality override.
- `driver.system_prompt_file`: repo-relative path to a backend system prompt file.
- `driver.model_provider`: optional alternate provider id for Codex-style backends.
- `driver.model_providers`: provider config map; if an entry declares `env_key`, EvE fails fast when that environment variable is missing.
- `driver.token_pricing`: per-driver pricing metadata override. Prefer shared pricing in `configs/pricing.yaml` unless a run needs a local override.

Tmux-only Codex fields (`codex_tmux`):

- `driver.approval_policy`: approval policy passed to the interactive backend.
- `driver.sandbox_mode`: sandbox mode passed to the interactive backend.
- `driver.allow_network`: when `sandbox_mode` is omitted, `true` maps to `danger-full-access`; otherwise the default is `workspace-write`.
- `driver.pool_size`: shared tmux pane pool size when any role uses a tmux backend. Default is the worker count.
- `driver.open_iterm2`: whether EvE opens the tmux session in a visible terminal window.
- `driver.completion_filename`: internal tmux completion marker filename.
- `driver.instruction_filename`: internal tmux instruction filename.

Claude-style backend fields (`claude_code`, `claude_code_tmux`) are backend-specific and not part of the public presets:

- `driver.effort_level`: native reasoning-effort spelling for these backends.
- `driver.setting_sources`: allowed settings sources; defaults to project/local.
- `driver.web_search`: `disabled` removes web tools for these backends.
- `driver.dangerously_skip_permissions`: applies to `claude_code_tmux`; the non-tmux `claude_code` builder bypasses permissions internally.
- Provider-policy fields such as `provider_base_url`, `api_key_env`, `policy_profile`, `allow_python_bash`, `allow_network`, `allow_subprocess`, and `allowed_env_vars` belong to legacy/provider-specific configs. Do not use them in public presets without a dedicated design task.

## Implementation References

This skill is grounded in the current runtime surfaces:

- `src/scaling_evolve/algorithms/eve/runtime/driver.py` for backend selection, role overrides, tmux pool creation, sandbox/web-search normalization, provider environment resolution, and pricing config loading.
- `src/scaling_evolve/providers/agent/drivers/codex_exec.py` for non-interactive Codex fields.
- `src/scaling_evolve/providers/agent/drivers/codex_tmux.py` for interactive Codex/tmux fields.
- `src/scaling_evolve/providers/agent/drivers/claude_code_tmux.py` and `src/scaling_evolve/providers/agent/config.py` for Claude-style backend fields.
