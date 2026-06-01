"""Default immutable renderer for Eve.

Uses the base README rendering and the fixed workflow-protocol entrypoint. This
is the renderer for any application that does not need an application-specific
entrypoint. Selected via:

```yaml
optimizer:
  immutable_renderer:
    _target_: scaling_evolve.algorithms.eve.workspace.immutable_renderers.default.DefaultRenderer
```
"""

from __future__ import annotations

from scaling_evolve.algorithms.eve.workspace.immutable_renderers.base import (
    ImmutableRenderer,
)


class DefaultRenderer(ImmutableRenderer):
    """Default renderer: base README rendering + the common entrypoint."""
