"""Driver construction for Eve runtimes."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from scaling_evolve.providers.agent.config import AgentProviderConfig
from scaling_evolve.providers.agent.drivers._metadata import TokenPricing, parse_token_pricing
from scaling_evolve.providers.agent.drivers.base import SessionDriver
from scaling_evolve.providers.agent.drivers.claude_code_tmux import (
    ClaudeCodeTmuxSessionDriver,
)
from scaling_evolve.providers.agent.drivers.codex_exec import CodexExecSessionDriver
from scaling_evolve.providers.agent.drivers.codex_tmux import (
    CodexTmuxPanePool,
    CodexTmuxSessionDriver,
)
from scaling_evolve.providers.agent.drivers.factory import build_claude_code_session_driver
from scaling_evolve.providers.agent.tmux_runtime import open_iterm2_window_for_session

_ROLE_NAMES = ("solver", "eval", "optimizer")


@dataclass(frozen=True)
class EveDriverSet:
    """Resolved role-specific drivers for one Eve run."""

    solver_driver: SessionDriver
    optimizer_driver: SessionDriver
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
    if _driver_name(role_cfg) == "codex_tmux":
        return _build_codex_tmux_driver(
            role_cfg,
            role=role,
            run_root=run_root,
            pane_pool=pane_pool,
            pricing_table=pricing_table,
        )
    if _driver_name(role_cfg) == "codex_exec":
        return _build_codex_exec_driver(
            role_cfg,
            role=role,
            run_root=run_root,
            pricing_table=pricing_table,
        )
    if _driver_name(role_cfg) == "claude_code_tmux":
        return _build_claude_code_tmux_driver(
            role_cfg,
            role=role,
            run_root=run_root,
            pane_pool=pane_pool,
            pricing_table=pricing_table,
        )

    normalized_cfg = dict(role_cfg)
    normalized_cfg.setdefault("kind", "agent_fork")
    normalized_cfg.setdefault("driver", "claude_code")
    normalized_cfg.pop("pool_size", None)
    normalized_cfg.pop("open_iterm2", None)
    normalized_cfg.pop("dangerously_skip_permissions", None)
    normalized_cfg.pop("web_search", None)
    normalized_cfg.pop("search_enabled", None)
    normalized_cfg["budget_prompt"] = _bool_config(
        role_cfg.get("budget_prompt"),
        default=True,
    )
    config = AgentProviderConfig.model_validate(normalized_cfg)
    disallowed_tools = _claude_disallowed_tools_from_driver_cfg(role_cfg)
    try:
        return build_claude_code_session_driver(
            config,
            disallowed_tools=disallowed_tools,
            pricing_table=pricing_table,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error


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
        _driver_name(_driver_cfg_for_role(driver_cfg, role_name))
        in {"codex_tmux", "claude_code_tmux"}
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
    optimizer_driver = build_driver(
        driver_cfg,
        role="optimizer",
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
        optimizer_driver=optimizer_driver,
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
        completion_filename=str(driver_cfg.get("completion_filename") or ".evolve-done.json"),
        instruction_filename=str(
            driver_cfg.get("instruction_filename") or ".evolve-instruction.md"
        ),
        timeout_seconds=float(driver_cfg.get("timeout_seconds") or 900.0),
        personality=cast(str | None, driver_cfg.get("personality")),
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
        timeout_seconds=float(driver_cfg.get("timeout_seconds") or 900.0),
        personality=cast(str | None, driver_cfg.get("personality")),
        role=role,
        web_search=_web_search_from_driver_cfg(driver_cfg),
        token_pricing=_token_pricing_from_driver_cfg(driver_cfg),
        pricing_table=pricing_table,
        model_provider=_string_config(driver_cfg.get("model_provider")),
        model_providers=_model_providers_from_driver_cfg(driver_cfg),
        provider_env=_provider_env_from_driver_cfg(driver_cfg),
    )


def _build_claude_code_tmux_driver(
    driver_cfg: dict[str, Any],
    *,
    role: str | None,
    run_root: str | Path | None,
    pane_pool: CodexTmuxPanePool | None,
    pricing_table: Mapping[str, TokenPricing] | None,
) -> ClaudeCodeTmuxSessionDriver:
    if pane_pool is None:
        if run_root is None:
            raise ValueError(
                "claude_code_tmux requires run_root so it can create a pane pool session."
            )
        pane_pool = CodexTmuxPanePool.create(
            session_name=_tmux_session_name(run_root),
            cwd=run_root,
            pane_count=_int_config(driver_cfg.get("pool_size"), default=1),
        )
        owns_pool = True
    else:
        owns_pool = False
    resolved_run_root = Path(run_root or pane_pool.cwd).expanduser().resolve()
    setting_sources = _claude_setting_sources_from_driver_cfg(driver_cfg)
    disallowed_tools = _claude_disallowed_tools_from_driver_cfg(driver_cfg)
    return ClaudeCodeTmuxSessionDriver(
        pane_pool=pane_pool,
        run_root=resolved_run_root,
        executable=str(driver_cfg.get("executable") or "claude"),
        model=_string_config(driver_cfg.get("model")),
        effort_level=_string_config(
            driver_cfg.get("effort_level") or driver_cfg.get("reasoning_effort")
        ),
        rollout_max_turns=_int_config(driver_cfg.get("rollout_max_turns"), default=200),
        budget_prompt=_bool_config(driver_cfg.get("budget_prompt"), default=True),
        timeout_seconds=float(driver_cfg.get("timeout_seconds") or 900.0),
        role=role,
        setting_sources=setting_sources or ("project", "local"),
        disallowed_tools=disallowed_tools,
        dangerously_skip_permissions=_bool_config(
            driver_cfg.get("dangerously_skip_permissions"),
            default=True,
        ),
        token_pricing=_token_pricing_from_driver_cfg(driver_cfg),
        pricing_table=pricing_table,
        owns_pool=owns_pool,
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


def _driver_name(driver_cfg: dict[str, Any]) -> str:
    raw_driver = driver_cfg.get("driver")
    if isinstance(raw_driver, str) and raw_driver:
        return raw_driver
    raw_provider = driver_cfg.get("provider")
    if isinstance(raw_provider, str) and raw_provider:
        return raw_provider
    return "claude_code"


def _int_config(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return max(value, 1)
    return default


def _bool_config(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


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


def _claude_setting_sources_from_driver_cfg(driver_cfg: dict[str, Any]) -> tuple[str, ...]:
    raw_setting_sources = driver_cfg.get("setting_sources")
    if isinstance(raw_setting_sources, list):
        setting_sources = tuple(
            str(source)
            for source in raw_setting_sources
            if str(source).strip() in {"user", "project", "local"}
        )
        if setting_sources:
            return setting_sources
    return ("project", "local")


def _claude_disallowed_tools_from_driver_cfg(driver_cfg: dict[str, Any]) -> tuple[str, ...]:
    if _web_search_from_driver_cfg(driver_cfg) == "disabled":
        return ("WebSearch", "WebFetch")
    return ()


def _string_config(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


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
