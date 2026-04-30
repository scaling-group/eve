"""Pure workspace strategy selection helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from scaling_evolve.core.bindings import AssessmentContract, TargetBinding
from scaling_evolve.core.common import JSONValue
from scaling_evolve.providers.agent.workspaces import (
    WorkspacePlan,
    WorkspacePurpose,
    WorkspaceStrategy,
)

MutationSurfaceInput = Mapping[str, JSONValue] | None


def resolve_workspace_plan(
    *,
    target_binding: TargetBinding,
    assessment: AssessmentContract,
    provider_kind: str,
    purpose: WorkspacePurpose,
    mutation_surface: MutationSurfaceInput = None,
    preferred_strategy: WorkspaceStrategy | None = None,
) -> WorkspacePlan:
    """Compile runtime inputs into a concrete workspace strategy."""

    strategy: WorkspaceStrategy
    if preferred_strategy is not None:
        strategy = preferred_strategy
    elif target_binding.default_workspace_strategy is not None:
        strategy = cast(WorkspaceStrategy, target_binding.default_workspace_strategy)
    elif purpose == "mutation":
        if _needs_full_workspace(assessment=assessment, purpose=purpose):
            strategy = "full_workspace"
        else:
            strategy = "artifact_only"
    elif _needs_full_workspace(assessment=assessment, purpose=purpose):
        strategy = "full_workspace"
    else:
        strategy = "artifact_only"

    roots = _surface_roots(mutation_surface)
    return WorkspacePlan(
        strategy=strategy,
        purpose=purpose,
        support_roots=list(dict.fromkeys(target_binding.support_roots + roots["read_roots"])),
        write_roots=roots["write_roots"],
        needs_git_semantics=purpose in {"evaluation", "review", "debug"}
        or bool(assessment.judge_commands),
    )


def _needs_full_workspace(
    *,
    assessment: AssessmentContract,
    purpose: WorkspacePurpose,
) -> bool:
    if purpose in {"review", "debug", "evaluation"}:
        return True
    return assessment.kind != "objective" or bool(assessment.judge_commands)


def _surface_roots(
    mutation_surface: MutationSurfaceInput,
) -> dict[str, list[str]]:
    if mutation_surface is None:
        return {"read_roots": [], "write_roots": []}
    mutation_surface_mapping = mutation_surface
    read_roots = mutation_surface_mapping.get("read_roots")
    write_roots = mutation_surface_mapping.get("write_roots")
    normalized_read_roots: list[str] = []
    normalized_write_roots: list[str] = []
    if isinstance(read_roots, list):
        for item in cast(list[object], read_roots):
            if isinstance(item, str):
                normalized_read_roots.append(item)
    if isinstance(write_roots, list):
        for item in cast(list[object], write_roots):
            if isinstance(item, str):
                normalized_write_roots.append(item)
    return {"read_roots": normalized_read_roots, "write_roots": normalized_write_roots}
