"""Portable lineage summary helpers."""

from __future__ import annotations

import hashlib

from scaling_evolve.core.common import JSONValue
from scaling_evolve.core.mutation import ProjectedState


def build_lineage_summary(projected_state: ProjectedState) -> dict[str, JSONValue]:
    """Build a stable portable summary from a projected state."""

    artifact_paths = [artifact.path for artifact in projected_state.artifacts if artifact.path]
    feedback = [
        {"source": item.source, "content": item.content} for item in projected_state.feedback
    ]
    return {
        "parent_node_id": projected_state.parent_node_id,
        "summary": projected_state.summary,
        "lineage_summary": projected_state.lineage_summary,
        "latest_score_summary": projected_state.latest_score_summary,
        "latest_failure_summary": projected_state.latest_failure_summary,
        "best_ancestor_summary": projected_state.best_ancestor_summary,
        "inspiration_summaries": list(projected_state.inspiration_summaries),
        "inherited_context": dict(projected_state.inherited_context),
        "artifact_paths": artifact_paths,
        "feedback": feedback,
        "metadata": dict(projected_state.metadata),
        "has_runtime_state": projected_state.runtime_state is not None,
        "has_portable_state": projected_state.portable_state is not None,
    }


def build_transcript_digest(transcript: str) -> dict[str, JSONValue]:
    """Build a stable digest summary for a transcript payload."""

    normalized = transcript.replace("\r\n", "\n")
    lines = normalized.splitlines()
    preview = [line for line in lines if line.strip()][:3]
    return {
        "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "line_count": len(lines),
        "preview": preview,
        "is_empty": normalized.strip() == "",
    }
