"""Agent execution-mode surface."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "InMemorySessionStore",
    "AgentProvider",
    "AgentProviderConfig",
    "SessionInfo",
    "SessionStore",
]


def __getattr__(name: str) -> object:
    if name == "AgentProvider":
        return import_module("scaling_evolve.providers.agent.backend").AgentProvider
    if name == "AgentProviderConfig":
        return import_module("scaling_evolve.providers.agent.config").AgentProviderConfig
    if name in {"InMemorySessionStore", "SessionInfo", "SessionStore"}:
        module = import_module("scaling_evolve.providers.agent.session_store")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
