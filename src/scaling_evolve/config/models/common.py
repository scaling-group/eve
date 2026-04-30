"""Common config model helpers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictConfigModel(BaseModel):
    """Shared strict model base for all config schema."""

    model_config = ConfigDict(extra="forbid", validate_default=True)
