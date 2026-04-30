"""Minimal opaque score helpers for Eve populations."""

from __future__ import annotations

import yaml
from optree import PyTree


def scalar(score: PyTree, *, preferred_key: str | None = None) -> float:
    """Project a score PyTree to a scalar or raise if no numeric view exists."""
    preferred_key_error = False
    if isinstance(score, dict) and preferred_key is not None:
        preferred_value = score.get(preferred_key)
        if isinstance(preferred_value, (int, float)) and not isinstance(preferred_value, bool):
            return float(preferred_value)
        preferred_key_error = True
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return float(score)
    if isinstance(score, dict):
        value = next(iter(score.values())) if len(score) == 1 else score.get("score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    if preferred_key_error:
        raise TypeError(
            f"score does not provide numeric preferred key {preferred_key!r}; "
            "expected that key, a scalar, a single-field mapping, or a mapping "
            "with numeric 'score'"
        )
    raise TypeError(
        "score must be a scalar, a single-field mapping, or a mapping with numeric 'score'"
    )


def score_block_lines(score: PyTree, *, indent: int = 0) -> list[str]:
    """Render a score PyTree as an indented YAML block without document markers."""
    prefix = " " * indent
    rendered = yaml.safe_dump(score, sort_keys=False).strip()
    return [f"{prefix}{line}" for line in rendered.splitlines()]
