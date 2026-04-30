"""Shared helpers for session-driver metadata and token pricing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from scaling_evolve.core.mutation import ProviderUsage


@dataclass(frozen=True)
class TokenPricing:
    """Per-million token pricing used for driver-side cost fallbacks."""

    input_per_million: float = 0.0
    output_per_million: float = 0.0
    cache_read_per_million: float = 0.0


def resolve_token_pricing(
    model: str | None,
    pricing: TokenPricing | None,
    pricing_table: Mapping[str, TokenPricing] | None = None,
) -> TokenPricing | None:
    """Resolve an explicit override or a lookup-table default for one model name."""

    if pricing is not None:
        return pricing
    if not isinstance(model, str) or not model.strip():
        return None
    if pricing_table is None:
        return None
    normalized_model = model.strip().lower()
    normalized_table = {
        str(key).strip().lower(): value
        for key, value in pricing_table.items()
        if isinstance(key, str) and key.strip()
    }

    exact = normalized_table.get(normalized_model)
    if exact is not None:
        return exact

    providerless_model = normalized_model.split("/", 1)[-1]
    exact_providerless = normalized_table.get(providerless_model)
    if exact_providerless is not None:
        return exact_providerless

    for key, value in sorted(normalized_table.items(), key=lambda item: len(item[0]), reverse=True):
        if normalized_model.startswith(key) or providerless_model.startswith(key):
            return value
        if key in normalized_model or key in providerless_model:
            return value
    return None


def parse_token_pricing(value: object) -> TokenPricing | None:
    """Parse a config-provided token_pricing override."""

    if value is None:
        return None
    if isinstance(value, TokenPricing):
        return value
    if not isinstance(value, Mapping):
        return None
    payload = {str(key): item for key, item in value.items()}
    return TokenPricing(
        input_per_million=_float_value(payload.get("input_per_million")),
        output_per_million=_float_value(payload.get("output_per_million")),
        cache_read_per_million=_float_value(payload.get("cache_read_per_million")),
    )


def compute_cost(usage: ProviderUsage, pricing: TokenPricing | None) -> float:
    """Return provider-reported cost when available, otherwise estimate from tokens."""

    if usage.model_cost_usd > 0:
        return float(usage.model_cost_usd)
    if pricing is None:
        return 0.0
    return (
        (usage.input_tokens * pricing.input_per_million) / 1_000_000.0
        + (usage.output_tokens * pricing.output_per_million) / 1_000_000.0
        + (usage.cache_read_tokens * pricing.cache_read_per_million) / 1_000_000.0
    )


def build_driver_execution_metadata(
    *,
    driver: str,
    command: Sequence[str],
    cwd: str | Path,
    exit_code: int | None = None,
    rollout_max_turns: int | None = None,
    timeout_seconds: float | None = None,
    model: str | None = None,
    effort_level: str | None = None,
    reasoning_effort: str | None = None,
    result_subtype: str | None = None,
    result_is_error: bool | None = None,
    accepted_partial_result: bool | None = None,
    num_turns: int | None = None,
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "driver": driver,
        "command": list(command),
        "cwd": str(cwd),
        "exit_code": exit_code,
        "rollout_max_turns": rollout_max_turns,
        "timeout_seconds": timeout_seconds,
        "model": model,
        "effort_level": effort_level,
        "reasoning_effort": reasoning_effort,
        "result_subtype": result_subtype,
        "result_is_error": result_is_error,
        "accepted_partial_result": accepted_partial_result,
        "num_turns": num_turns,
    }
    payload.update(extra)
    return {key: value for key, value in payload.items() if value is not None}


def _float_value(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0
