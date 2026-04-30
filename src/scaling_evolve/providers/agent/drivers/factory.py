"""Shared builders for agent session drivers."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from scaling_evolve.providers.agent.config import AgentProviderConfig
from scaling_evolve.providers.agent.drivers._metadata import TokenPricing
from scaling_evolve.providers.agent.drivers.claude_code import ClaudeCodeSessionDriver


def build_claude_code_session_driver(
    config: AgentProviderConfig,
    *,
    system_prompt_append: str | None = None,
    disallowed_tools: Sequence[str] = (),
    pricing_table: Mapping[str, TokenPricing] | None = None,
) -> ClaudeCodeSessionDriver:
    """Build a Claude Code session driver from provider config."""

    provider_auth_override_configured = any(
        value is not None
        for value in (
            config.provider_base_url,
            config.api_key_env,
        )
    )
    auth_token: str | None = None
    if provider_auth_override_configured:
        if config.api_key_env is None:
            raise ValueError("Session provider overrides require `mutation.provider.api_key_env`.")
        auth_token = os.environ.get(config.api_key_env)
        if not auth_token:
            raise ValueError(f"Missing required environment variable `{config.api_key_env}`.")

    return ClaudeCodeSessionDriver(
        executable=config.executable,
        provider_base_url=config.provider_base_url,
        api_key_env=config.api_key_env,
        auth_token=auth_token,
        model=config.model,
        dangerously_skip_permissions=True,
        disallowed_tools=disallowed_tools,
        isolate_config=config.isolate_config,
        temperature=config.temperature,
        rollout_max_turns=config.rollout_max_turns,
        budget_prompt=config.budget_prompt,
        timeout_seconds=config.timeout_seconds,
        context_window=config.context_window,
        auto_compact_pct=config.auto_compact_pct,
        max_output_tokens=config.max_output_tokens,
        max_thinking_tokens=config.max_thinking_tokens,
        effort_level=config.effort_level,
        disable_adaptive_thinking=config.disable_adaptive_thinking,
        token_pricing=config.token_pricing,
        pricing_table=pricing_table,
        setting_sources=config.setting_sources or ["local"],
        inherited_config_dir=os.environ.get("CLAUDE_CONFIG_DIR"),
        openrouter_provider_pin=config.openrouter_provider_pin,
        system_prompt_append=system_prompt_append,
    )
