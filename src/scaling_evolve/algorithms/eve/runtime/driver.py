"""Driver construction for Eve runtimes."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from scaling_evolve.providers.agent.drivers._metadata import TokenPricing, parse_token_pricing
from scaling_evolve.providers.agent.drivers.base import SessionDriver
from scaling_evolve.providers.agent.drivers.codex_exec import CodexExecSessionDriver
from scaling_evolve.providers.agent.drivers.codex_tmux import (
    CodexTmuxPanePool,
    CodexTmuxSessionDriver,
)
from scaling_evolve.providers.agent.tmux_runtime import open_iterm2_window_for_session

_ROLE_NAMES = ("solver", "eval")
_SUPPORTED_DRIVER_NAMES = ("codex_exec", "codex_tmux")

# Repo root, used to resolve repo-relative asset paths in driver config (e.g.
# `system_prompt_file`) the same way application assets are resolved.
_REPO_ROOT = Path(__file__).resolve().parents[5]


@dataclass(frozen=True)
class EveDriverSet:
    """Resolved role-specific drivers for one Eve run."""

    solver_driver: SessionDriver
    eval_driver_factory: Callable[[], SessionDriver]
    pane_pool: CodexTmuxPanePool | None = None

    def close(self) -> None:
        if self.pane_pool is not None:
            self.pane_pool.close()


def build_driver(
    driver_cfg: dict[str, Any],
    *,
    role: str | None = None,
    run_root: str | Path | None = None,
    pane_pool: CodexTmuxPanePool | None = None,
    pricing_table: Mapping[str, TokenPricing] | None = None,
) -> SessionDriver:
    role_cfg = _driver_cfg_for_role(driver_cfg, role)
    driver_name = _driver_name(role_cfg)
    if driver_name == "codex_tmux":
        return _build_codex_tmux_driver(
            role_cfg,
            role=role,
            run_root=run_root,
            pane_pool=pane_pool,
            pricing_table=pricing_table,
        )
    if driver_name == "codex_exec":
        return _build_codex_exec_driver(
            role_cfg,
            role=role,
            run_root=run_root,
            pricing_table=pricing_table,
        )
    if driver_name is None:
        raise SystemExit(
            "Missing driver config. Set `driver.driver` or `driver.provider` to "
            "`codex_exec` or `codex_tmux`."
        )
    raise SystemExit(
        f"Unsupported driver `{driver_name}`. Supported drivers: "
        f"{', '.join(_SUPPORTED_DRIVER_NAMES)}."
    )


def build_driver_factory(
    driver_cfg: dict[str, Any],
    *,
    role: str | None = None,
    run_root: str | Path | None = None,
    pane_pool: CodexTmuxPanePool | None = None,
    pricing_table: Mapping[str, TokenPricing] | None = None,
) -> Callable[[], SessionDriver]:
    snapshot = dict(driver_cfg)
    return lambda: build_driver(
        dict(snapshot),
        role=role,
        run_root=run_root,
        pane_pool=pane_pool,
        pricing_table=pricing_table,
    )


def build_role_drivers(
    driver_cfg: dict[str, Any],
    *,
    run_root: str | Path,
    workers: int,
    pricing_table: Mapping[str, TokenPricing] | None = None,
) -> EveDriverSet:
    pane_pool: CodexTmuxPanePool | None = None
    if any(
        _driver_name(_driver_cfg_for_role(driver_cfg, role_name)) == "codex_tmux"
        for role_name in _ROLE_NAMES
    ):
        pool_size = _int_config(driver_cfg.get("pool_size"), default=workers)
        pane_pool = CodexTmuxPanePool.create(
            session_name=_tmux_session_name(run_root),
            cwd=Path(run_root).expanduser().resolve(),
            pane_count=pool_size,
        )
        if _bool_config(driver_cfg.get("open_iterm2"), default=True):
            open_iterm2_window_for_session(pane_pool.session_name)

    solver_driver = build_driver(
        driver_cfg,
        role="solver",
        run_root=run_root,
        pane_pool=pane_pool,
        pricing_table=pricing_table,
    )
    eval_driver_factory = build_driver_factory(
        driver_cfg,
        role="eval",
        run_root=run_root,
        pane_pool=pane_pool,
        pricing_table=pricing_table,
    )
    return EveDriverSet(
        solver_driver=solver_driver,
        eval_driver_factory=eval_driver_factory,
        pane_pool=pane_pool,
    )


def _build_codex_tmux_driver(
    driver_cfg: dict[str, Any],
    *,
    role: str | None,
    run_root: str | Path | None,
    pane_pool: CodexTmuxPanePool | None,
    pricing_table: Mapping[str, TokenPricing] | None,
) -> CodexTmuxSessionDriver:
    if pane_pool is None:
        if run_root is None:
            raise ValueError("codex_tmux requires run_root so it can create a pane pool session.")
        pane_pool = CodexTmuxPanePool.create(
            session_name=_tmux_session_name(run_root),
            cwd=run_root,
            pane_count=_int_config(driver_cfg.get("pool_size"), default=1),
        )
        owns_pool = True
    else:
        owns_pool = False
    resolved_run_root = Path(run_root or pane_pool.cwd).expanduser().resolve()
    return CodexTmuxSessionDriver(
        pane_pool=pane_pool,
        run_root=resolved_run_root,
        executable=str(driver_cfg.get("executable") or "codex"),
        model=str(driver_cfg.get("model") or "gpt-5.4-mini"),
        reasoning_effort=str(
            driver_cfg.get("reasoning_effort") or driver_cfg.get("effort_level") or "low"
        ),
        rollout_max_turns=_int_config(driver_cfg.get("rollout_max_turns"), default=200),
        budget_prompt=_bool_config(driver_cfg.get("budget_prompt"), default=True),
        enable_multi_agent=_optional_bool_config(driver_cfg.get("enable_multi_agent")),
        completion_filename=str(driver_cfg.get("completion_filename") or ".evolve-done.json"),
        instruction_filename=str(
            driver_cfg.get("instruction_filename") or ".evolve-instruction.md"
        ),
        timeout_seconds=float(driver_cfg.get("timeout_seconds") or 900.0),
        personality=cast(str | None, driver_cfg.get("personality")),
        codex_system_prompt_file=_resolve_repo_relative_path(driver_cfg.get("system_prompt_file")),
        role=role,
        approval_policy=str(driver_cfg.get("approval_policy") or "never"),
        sandbox_mode=str(_sandbox_mode_from_driver_cfg(driver_cfg)),
        web_search=_web_search_from_driver_cfg(driver_cfg),
        token_pricing=_token_pricing_from_driver_cfg(driver_cfg),
        pricing_table=pricing_table,
        model_provider=_string_config(driver_cfg.get("model_provider")),
        model_providers=_model_providers_from_driver_cfg(driver_cfg),
        provider_env=_provider_env_from_driver_cfg(driver_cfg),
        owns_pool=owns_pool,
    )


def _build_codex_exec_driver(
    driver_cfg: dict[str, Any],
    *,
    role: str | None,
    run_root: str | Path | None,
    pricing_table: Mapping[str, TokenPricing] | None,
) -> CodexExecSessionDriver:
    resolved_run_root = Path(run_root or ".").expanduser().resolve()
    return CodexExecSessionDriver(
        run_root=resolved_run_root,
        executable=str(driver_cfg.get("executable") or "codex"),
        model=str(driver_cfg.get("model") or "gpt-5.4-mini"),
        reasoning_effort=str(
            driver_cfg.get("reasoning_effort") or driver_cfg.get("effort_level") or "low"
        ),
        rollout_max_turns=_int_config(driver_cfg.get("rollout_max_turns"), default=200),
        budget_prompt=_bool_config(driver_cfg.get("budget_prompt"), default=True),
        enable_multi_agent=_optional_bool_config(driver_cfg.get("enable_multi_agent")),
        timeout_seconds=float(driver_cfg.get("timeout_seconds") or 900.0),
        personality=cast(str | None, driver_cfg.get("personality")),
        codex_system_prompt_file=_resolve_repo_relative_path(driver_cfg.get("system_prompt_file")),
        role=role,
        web_search=_web_search_from_driver_cfg(driver_cfg),
        token_pricing=_token_pricing_from_driver_cfg(driver_cfg),
        pricing_table=pricing_table,
        model_provider=_string_config(driver_cfg.get("model_provider")),
        model_providers=_model_providers_from_driver_cfg(driver_cfg),
        provider_env=_provider_env_from_driver_cfg(driver_cfg),
    )


def _driver_cfg_for_role(driver_cfg: dict[str, Any], role: str | None) -> dict[str, Any]:
    resolved = {key: value for key, value in driver_cfg.items() if key != "overrides"}
    if role is None:
        return resolved
    overrides = driver_cfg.get("overrides")
    if not isinstance(overrides, dict):
        return resolved
    role_override = overrides.get(role)
    if isinstance(role_override, dict):
        resolved.update(role_override)
    return resolved


def _token_pricing_from_driver_cfg(driver_cfg: dict[str, Any]):
    return parse_token_pricing(driver_cfg.get("token_pricing"))


def load_pricing_table(path: str | Path) -> dict[str, TokenPricing]:
    """Load a token-pricing lookup table from YAML."""

    pricing_path = Path(path).expanduser().resolve()
    if not pricing_path.exists():
        raise FileNotFoundError(f"Pricing config not found: {pricing_path}")
    payload = yaml.safe_load(pricing_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Pricing config must be a mapping: {pricing_path}")

    table: dict[str, TokenPricing] = {}
    for raw_key, raw_value in payload.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            continue
        pricing = parse_token_pricing(raw_value)
        if pricing is None:
            raise ValueError(f"Invalid pricing entry for `{raw_key}` in {pricing_path}")
        table[raw_key.strip()] = pricing
    return table


def _driver_name(driver_cfg: dict[str, Any]) -> str | None:
    raw_driver = driver_cfg.get("driver")
    if isinstance(raw_driver, str) and raw_driver:
        return raw_driver
    raw_provider = driver_cfg.get("provider")
    if isinstance(raw_provider, str) and raw_provider:
        return raw_provider
    return None


def _int_config(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return max(value, 1)
    return default


def _bool_config(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _optional_bool_config(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _sandbox_mode_from_driver_cfg(driver_cfg: dict[str, Any]) -> str:
    configured = driver_cfg.get("sandbox_mode")
    if isinstance(configured, str) and configured:
        return configured
    if _bool_config(driver_cfg.get("allow_network"), default=False):
        return "danger-full-access"
    return "workspace-write"


def _web_search_from_driver_cfg(driver_cfg: dict[str, Any]) -> str:
    configured = driver_cfg.get("web_search")
    if isinstance(configured, str):
        normalized = configured.strip().lower()
        if normalized in {"disabled", "live", "cached"}:
            return normalized
    return "live" if _bool_config(driver_cfg.get("search_enabled"), default=False) else "disabled"


def _string_config(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _resolve_repo_relative_path(value: object) -> str | None:
    """Resolve a config path to an absolute string (None passes through).

    Relative paths anchor to the repo root, like application asset paths.
    """
    text = _string_config(value)
    if text is None:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return str(path.resolve())


def _model_providers_from_driver_cfg(driver_cfg: dict[str, Any]) -> dict[str, dict[str, object]]:
    raw = driver_cfg.get("model_providers")
    if not isinstance(raw, dict):
        return {}
    providers: dict[str, dict[str, object]] = {}
    for provider_id, provider_cfg in raw.items():
        if not isinstance(provider_id, str) or not provider_id.strip():
            continue
        if not isinstance(provider_cfg, dict):
            continue
        providers[provider_id] = {
            str(key): value for key, value in provider_cfg.items() if value is not None
        }
    return providers


def _provider_env_from_driver_cfg(driver_cfg: dict[str, Any]) -> dict[str, str]:
    provider_env: dict[str, str] = {}
    for provider_cfg in _model_providers_from_driver_cfg(driver_cfg).values():
        env_key = _string_config(provider_cfg.get("env_key"))
        if env_key is not None:
            env_value = os.environ.get(env_key)
            if not env_value:
                raise SystemExit(f"Missing required environment variable `{env_key}`.")
            provider_env[env_key] = env_value
    return provider_env


def _tmux_session_name(run_root: str | Path) -> str:
    root = Path(run_root).expanduser().resolve()
    safe_name = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-" for char in root.name
    ).strip("-")
    suffix = safe_name[-24:] or "eve"
    return f"eve-{suffix}"
