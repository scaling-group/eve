"""Snapshot storage protocol surface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from scaling_evolve.core.common import JSONValue
from scaling_evolve.core.engine import PortableStateRef, RuntimeStateRef


@runtime_checkable
class SnapshotStore(Protocol):
    """Protocol for persisting runtime and portable snapshots."""

    def save_runtime_state(
        self,
        ref: RuntimeStateRef,
        payload: Mapping[str, JSONValue],
    ) -> None:
        """Persist runtime-owned state."""

        ...

    def save_portable_state(
        self,
        ref: PortableStateRef,
        payload: Mapping[str, JSONValue],
    ) -> None:
        """Persist portable state."""

        ...

    def load_portable_state(self, ref: PortableStateRef) -> Mapping[str, JSONValue]:
        """Load previously stored portable state."""

        ...


@runtime_checkable
class ReadableSnapshotStore(SnapshotStore, Protocol):
    """Sibling protocol that can read both portable and runtime metadata."""

    def load_runtime_state(self, ref: RuntimeStateRef) -> Mapping[str, JSONValue]:
        """Load previously stored runtime state metadata."""

        ...
