"""Config model package."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AgentProviderConfig": "scaling_evolve.providers.agent.config",
    "StrictConfigModel": "scaling_evolve.config.models.common",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
