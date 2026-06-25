"""No-op immutable-asset renderer."""

from __future__ import annotations

from scaling_evolve.algorithms.eve.workspace.immutable_renderers.base import (
    ImmutableRenderer,
)


class StaticRenderer(ImmutableRenderer):
    """Land immutable templates verbatim.

    ``DefaultRenderer`` substitutes solver markers into a README template and
    fails validation when they are missing. Judge rubrics are static text with
    no markers, so they use this renderer to land unchanged.
    """

    def render(self, template: str, ctx: object | None = None) -> str:
        _ = ctx
        return template
