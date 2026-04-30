"""Target-binding and assessment models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from scaling_evolve.core.common import DomainModel


class TargetBinding(DomainModel):
    """Resolved target binding for an application."""

    kind: Literal["managed"] = "managed"
    repo_root: str | None = None
    config_root: str | None = None
    support_roots: list[str] = Field(default_factory=list)
    default_workspace_strategy: str | None = None


class AssessmentContract(DomainModel):
    """Resolved assessment contract for an application."""

    kind: Literal["objective", "weak_judge", "manual"]
    judge_commands: list[str] = Field(default_factory=list)
    primary_artifacts: list[str] = Field(default_factory=list)
