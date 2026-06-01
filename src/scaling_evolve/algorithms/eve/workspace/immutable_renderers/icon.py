"""ICON immutable renderer for Eve.

Same README rendering as the base, but the entrypoint appends ICON's
application-specific stop rule. This is the renderer for the ICON application;
the application-specific entrypoint is expressed as a subclass rather than a
config string or a file in the immutable tree. Selected via:

```yaml
optimizer:
  immutable_renderer:
    _target_: scaling_evolve.algorithms.eve.workspace.immutable_renderers.icon.IconRenderer
```
"""

from __future__ import annotations

from scaling_evolve.algorithms.eve.workspace.immutable_renderers.base import (
    ImmutableRenderer,
)

_ICON_STOP_RULE = """=================VERY IMPORTANT RULE BELOW!!!=====================================
Don't keep improving forever. Once you think code is runnable and `check-runner` passed, STOP IMMEDIATELY, but only after one of these is true:
1. you made at least one research-bearing delta that is meaningfully distinct from the copied prefill solver, or
2. you explicitly recorded in `logs/optimize/session.md` why no such delta is justified in this workspace.
A validation-only repair such as checkpoint materialization hygiene does not by itself satisfy rule 1.
Never run evaluation in any sense by yourself, or perform a eval-edit loop. End the run and leave the evaluation and improvement to the future!
Before executing, repeat the above rule 3 times by yourself to make sure you have it in mind!!! Make sure you always follow it!!!
===================VERY IMPORTANT RULE ABOVE!!!==================================="""


class IconRenderer(ImmutableRenderer):
    """ICON renderer: base README rendering + the common entrypoint plus ICON's stop rule."""

    ENTRYPOINT: str = f"{ImmutableRenderer.ENTRYPOINT}\n\n{_ICON_STOP_RULE}"
