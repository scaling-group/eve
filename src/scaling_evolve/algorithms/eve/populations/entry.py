"""Shared population entry type for Eve."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from optree import PyTree

ID_HEX_LENGTH = 12


@dataclass
class PopulationEntry:
    """A single entry in a population (task or optimizer).

    Attributes:
        id: unique identifier.
        files: relative path -> file content (free-form multi-file artifact).
        score: opaque score pytree stored and restored without interpretation.
        logs: relative path -> content (free-form log directory). Binary files
            are stored via a portable encoded string envelope.
    """

    id: str
    files: dict[str, str]
    score: PyTree
    logs: dict[str, str]  # relative path -> content or portable encoded binary payload

    @staticmethod
    def make_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:ID_HEX_LENGTH]}"
