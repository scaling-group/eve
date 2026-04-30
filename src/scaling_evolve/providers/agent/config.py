"""Config for persistent session mutation providers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import Field, field_validator, model_validator

from scaling_evolve.config.models.common import StrictConfigModel
from scaling_evolve.providers.agent.drivers._metadata import TokenPricing

_LOGGER = logging.getLogger(__name__)
_SEEN_LEGACY_WARNINGS: set[str] = set()


def _warn_legacy_once(message: str) -> None:
    if message in _SEEN_LEGACY_WARNINGS:
        return
    _SEEN_LEGACY_WARNINGS.add(message)
    _LOGGER.warning(message)


def _default_allowed_tools() -> list[str]:
    return []


def _default_visible_tools() -> list[str]:
    return []


@dataclass(frozen=True)
class _ProviderDefaults:
    api_key_env: str
    llm_base_url: str
    session_base_url: str
    max_output_tokens: int
    context_window: int | None = None


_PROVIDER_DEFAULTS: dict[str, _ProviderDefaults] = {
    "deepseek": _ProviderDefaults(
        api_key_env="DEEPSEEK_API_KEY",
        llm_base_url="https://api.deepseek.com",
        session_base_url="https://api.deepseek.com/anthropic",
        max_output_tokens=8192,
        context_window=131072,
    ),
    "openrouter": _ProviderDefaults(
        api_key_env="OPENROUTER_API_KEY",
        llm_base_url="https://openrouter.ai/api/v1",
        session_base_url="https://openrouter.ai/api",
        max_output_tokens=65536,
        context_window=196608,
    ),
}

_POLICY_PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "benchmark_safe": {
        "allow_python_bash": True,
        "allow_network": False,
        "allow_subprocess": False,
        "allowed_env_vars": [],
    },
    "open_agent": {
        "allow_python_bash": True,
        "allow_network": True,
        "allow_subprocess": True,
        "allowed_env_vars": None,
    },
}

_PYTHON_BASH_PATTERNS = ("Bash(python *)", "Bash(python3 *)")


def _mapping_to_str_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    raw_payload = cast(Mapping[object, object], value)
    return {str(raw_key): raw_item for raw_key, raw_item in raw_payload.items()}


def _provider_defaults(provider: object) -> _ProviderDefaults | None:
    if not isinstance(provider, str):
        return None
    return _PROVIDER_DEFAULTS.get(provider)


def _infer_session_provider(payload: Mapping[str, object]) -> str | None:
    provider_base_url = payload.get("provider_base_url")
    api_key_env = payload.get("api_key_env")
    for provider, defaults in _PROVIDER_DEFAULTS.items():
        if provider_base_url == defaults.session_base_url or api_key_env == defaults.api_key_env:
            return provider
    return None


class AgentProviderConfig(StrictConfigModel):
    """Config for persistent session providers."""

    kind: Literal["agent_fork"]
    driver: Literal["claude_code", "codex_cli", "codex_tmux", "claude_code_tmux", "codex_exec"]
    provider: Literal["deepseek", "openrouter"] | None = None
    executable: str = "claude"
    permission_mode: Literal[
        "default",
        "acceptEdits",
        "plan",
        "dontAsk",
        "bypassPermissions",
    ] = Field(
        default="bypassPermissions",
        description=(
            "Legacy compatibility field for older YAML configs. Session drivers ignore "
            "this value and manage permission bypass internally."
        ),
    )
    allowed_tools: list[str] = Field(default_factory=_default_allowed_tools)
    visible_tools: list[str] = Field(default_factory=_default_visible_tools)
    isolate_config: bool = Field(
        default=False,
        description=(
            "When True, redirect CLAUDE_CONFIG_DIR to a per-workspace scratch directory. "
            "This no longer implies --bare; Claude Code still loads the configured "
            "setting sources and workspace hooks."
        ),
    )
    provider_base_url: str | None = None
    openrouter_provider_pin: str | None = None
    api_key_env: str | None = None
    model: str | None = None
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    rollout_max_turns: int = Field(
        default=200,
        gt=0,
        description=(
            "Maximum turns per rollout (one spawn or resume call). "
            "Infrastructure-level cap only: workflow code may call resume() multiple "
            "times, so aggregate session turns can exceed this value. "
            "If you need a hard session-level cap, enforce it at the workflow layer."
        ),
    )
    budget_prompt: bool = Field(
        default=True,
        description=(
            "Enable BudgetPrompt injection via hooks. "
            "When True (default), the agent is told about its turn budget and "
            "sees remaining turns after every turn. Set to False for baseline "
            "experiments where you want hard enforcement without the agent knowing."
        ),
    )
    timeout_seconds: float = Field(default=900.0, gt=0.0)
    fork_mode: Literal["native"] = "native"
    fallback_mode: Literal["summary_only"] = "summary_only"
    preferred_workspace_strategy: Literal["artifact_only", "full_workspace"] | None = None
    context_window: int | None = None
    auto_compact_pct: int | None = None
    max_output_tokens: int | None = None
    max_thinking_tokens: int | None = None
    effort_level: Literal["auto", "low", "medium", "high", "max"] | None = None
    disable_adaptive_thinking: bool | None = None
    token_pricing: TokenPricing | None = None
    setting_sources: list[Literal["user", "project", "local"]] = Field(
        default_factory=lambda: ["project", "local"]
    )
    policy_profile: Literal["benchmark_safe", "open_agent"] | None = None
    allow_python_bash: bool | None = None
    allow_network: bool | None = None
    allow_subprocess: bool | None = None
    allowed_env_vars: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _apply_provider_defaults(cls, value: object) -> object:
        payload = _mapping_to_str_dict(value)
        if payload is None:
            return value
        provider = payload.get("provider")
        if provider is None:
            inferred_provider = _infer_session_provider(payload)
            if inferred_provider is not None:
                payload["provider"] = inferred_provider
                provider = inferred_provider
        profile = payload.get("policy_profile")
        if isinstance(profile, str):
            for key, default in _POLICY_PROFILE_DEFAULTS.get(profile, {}).items():
                payload.setdefault(key, default)
        defaults = _provider_defaults(provider)
        if defaults is not None:
            payload.setdefault("api_key_env", defaults.api_key_env)
            payload.setdefault("provider_base_url", defaults.session_base_url)
            if defaults.context_window is not None:
                payload.setdefault("context_window", defaults.context_window)
            payload.setdefault("max_output_tokens", defaults.max_output_tokens)
        raw_allowed_tools = payload.get("allowed_tools")
        if isinstance(raw_allowed_tools, list):
            allowed_tools = [
                item for item in raw_allowed_tools if isinstance(item, str) and item.strip()
            ]
            allow_python_bash = payload.get("allow_python_bash")
            if allow_python_bash is True:
                for pattern in _PYTHON_BASH_PATTERNS:
                    if pattern not in allowed_tools:
                        allowed_tools.append(pattern)
            elif allow_python_bash is False:
                allowed_tools = [
                    item for item in allowed_tools if item not in _PYTHON_BASH_PATTERNS
                ]
        else:
            allowed_tools = []
        payload["allowed_tools"] = allowed_tools
        return payload

    @field_validator("allowed_env_vars", mode="before")
    @classmethod
    def _normalize_allowed_env_vars(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, list):
            return value
        seen: set[str] = set()
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            pattern = item.strip()
            if not pattern or pattern in seen:
                continue
            seen.add(pattern)
            normalized.append(pattern)
        return normalized

    @field_validator("fallback_mode", mode="before")
    @classmethod
    def _normalize_fallback_mode(cls, value: object) -> object:
        if value == "rehydrate":
            _warn_legacy_once("Legacy fallback mode 'rehydrate' mapped to 'summary_only'.")
            return "summary_only"
        return value
