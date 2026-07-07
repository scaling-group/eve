"""Config for persistent session mutation providers."""

from __future__ import annotations

import logging
from typing import Literal

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


_POLICY_PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "benchmark_safe": {
        "allow_network": False,
        "allow_subprocess": False,
        "allowed_env_vars": [],
    },
    "open_agent": {
        "allow_network": True,
        "allow_subprocess": True,
        "allowed_env_vars": None,
    },
}


def _mapping_to_str_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return {str(raw_key): raw_item for raw_key, raw_item in value.items()}


class AgentProviderConfig(StrictConfigModel):
    """Config for persistent session providers."""

    kind: Literal["agent_fork"]
    driver: Literal["codex_tmux", "codex_exec"]
    executable: str = "codex"
    model: str | None = None
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
    enable_multi_agent: bool | None = Field(
        default=None,
        description=(
            "Optional Codex CLI feature override. None preserves the CLI default; "
            "True passes features.multi_agent=true; False passes "
            "features.multi_agent=false."
        ),
    )
    timeout_seconds: float = Field(default=900.0, gt=0.0)
    fork_mode: Literal["native"] = "native"
    fallback_mode: Literal["summary_only"] = "summary_only"
    preferred_workspace_strategy: Literal["artifact_only", "full_workspace"] | None = None
    token_pricing: TokenPricing | None = None
    policy_profile: Literal["benchmark_safe", "open_agent"] | None = None
    allow_network: bool | None = None
    allow_subprocess: bool | None = None
    allowed_env_vars: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _apply_policy_defaults(cls, value: object) -> object:
        payload = _mapping_to_str_dict(value)
        if payload is None:
            return value
        profile = payload.get("policy_profile")
        if isinstance(profile, str):
            for key, default in _POLICY_PROFILE_DEFAULTS.get(profile, {}).items():
                payload.setdefault(key, default)
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
            _warn_legacy_once("Fallback mode 'rehydrate' mapped to 'summary_only'.")
            return "summary_only"
        return value
