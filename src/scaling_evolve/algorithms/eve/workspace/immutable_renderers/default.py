"""Default immutable renderer for Eve.

Uses the base README rendering and the prompt-folder entrypoint text. Selected via:

```yaml
optimizer:
  prompt: configs/eve/optimizer/circle_packing/prompt
  immutable_renderer:
    _target_: scaling_evolve.algorithms.eve.workspace.immutable_renderers.default.DefaultRenderer
```
"""

from __future__ import annotations

from scaling_evolve.algorithms.eve.workspace.immutable_renderers.base import (
    ImmutableRenderer,
)


class DefaultRenderer(ImmutableRenderer):
    """Default renderer: base README rendering + configured entrypoint text."""
