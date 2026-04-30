"""Artifact storage protocol surface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from scaling_evolve.core.common import JSONValue
from scaling_evolve.core.engine import ArtifactRef
from scaling_evolve.core.enums import ArtifactKind


@runtime_checkable
class ArtifactStore(Protocol):
    """Protocol for artifact persistence."""

    def put_text(
        self,
        kind: ArtifactKind,
        text: str,
        *,
        filename: str | None = None,
    ) -> ArtifactRef:
        """Persist a text artifact."""

        ...

    def put_json(
        self,
        kind: ArtifactKind,
        payload: Mapping[str, JSONValue],
        *,
        filename: str | None = None,
    ) -> ArtifactRef:
        """Persist a JSON artifact."""

        ...

    def read_text(self, ref: ArtifactRef) -> str:
        """Read a text artifact."""

        ...
