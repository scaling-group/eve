"""Shared core-domain primitives."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict

NodeId: TypeAlias = str
RunId: TypeAlias = str
EdgeId: TypeAlias = str
ArtifactId: TypeAlias = str
JSONValue: TypeAlias = Any
_SEEN_LEGACY_WARNINGS: set[str] = set()


def _legacy_name(*parts: str) -> str:
    return "_".join(parts)


class DomainModel(BaseModel):
    """Shared strict base model for protocol-facing domain types."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def warn_legacy_once(message: str, *, logger: logging.Logger) -> None:
    """Emit one warning per normalized legacy payload shape."""

    if message in _SEEN_LEGACY_WARNINGS:
        return
    _SEEN_LEGACY_WARNINGS.add(message)
    logger.warning(message)


def _parse_timestamp(value: object) -> datetime | None:
    """Parse one possibly-legacy timestamp value into UTC."""

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def normalize_provider_kind(
    value: object,
    *,
    logger: logging.Logger | None = None,
) -> object:
    """Normalize legacy provider names to the canonical provider kind."""

    if value == _legacy_name("single", "call"):
        if logger is not None:
            warn_legacy_once(
                f"Legacy provider kind '{_legacy_name('single', 'call')}' mapped to 'llm'.",
                logger=logger,
            )
        return "llm"
    if value == _legacy_name("session", "fork"):
        if logger is not None:
            warn_legacy_once(
                f"Legacy provider kind '{_legacy_name('session', 'fork')}' mapped to 'agent_fork'.",
                logger=logger,
            )
        return "agent_fork"
    if value == "session":
        if logger is not None:
            warn_legacy_once(
                "Legacy provider kind 'session' mapped to 'agent_fork'.",
                logger=logger,
            )
        return "agent_fork"
    if value == "agent_sdk":
        if logger is not None:
            warn_legacy_once(
                "Legacy provider kind 'agent_sdk' mapped to 'agent_fork'.",
                logger=logger,
            )
        return "agent_fork"
    return value


def primary_score(node: object, *, missing: float = float("-inf")) -> float:
    """Extract a comparable primary score from node-like objects."""

    candidate_score = getattr(node, "primary_score", None)
    if isinstance(candidate_score, int | float):
        return float(candidate_score)

    raw_score = getattr(node, "score", None)
    if isinstance(raw_score, int | float):
        return float(raw_score)

    value = getattr(raw_score, "primary_score", None)
    if isinstance(value, int | float):
        return float(value)
    return missing
