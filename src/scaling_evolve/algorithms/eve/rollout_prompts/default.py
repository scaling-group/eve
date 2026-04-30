"""Hydra-instantiable default rollout prompt objects for Eve."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Protocol


def _read_prompt_text(path: str) -> str:
    return (
        files("scaling_evolve.algorithms.eve.rollout_prompts")
        .joinpath(*path.split("/"))
        .read_text(encoding="utf-8")
        .strip()
    )


@dataclass(frozen=True)
class PromptContext:
    workspace: Path
    transcript_path: Path | None = None
    rollout_max_turns: int | None = None
    turns_remaining: int | None = None
    prompt: str | None = None
    turn_number: int | None = None


class RolloutPrompt(Protocol):
    """Interface contract for rollout-time prompt injection."""

    def system(self, ctx: PromptContext) -> str | None: ...

    def user(self, ctx: PromptContext) -> str | None: ...

    def turn(self, ctx: PromptContext) -> str | None: ...


class _BaseRolloutPrompt:
    """Shared .md-loading machinery for template-driven rollout prompts."""

    def __init__(
        self,
        *,
        system_files: tuple[str, ...] = (),
        user_files: tuple[str, ...] = (),
        turn_files: tuple[str, ...] = (),
    ) -> None:
        self._system_template = self._compose(system_files)
        self._user_template = self._compose(user_files)
        self._turn_template = self._compose(turn_files)

    @staticmethod
    def _compose(paths: tuple[str, ...]) -> str:
        if not paths:
            return ""
        parts = [_read_prompt_text(path) for path in paths]
        return "\n\n".join(part for part in parts if part).strip()

    def system(self, ctx: PromptContext) -> str | None:
        if not self._system_template:
            return None
        return self._render(self._system_template, ctx)

    def user(self, ctx: PromptContext) -> str | None:
        if not self._user_template:
            return None
        return self._render(self._user_template, ctx)

    def turn(self, ctx: PromptContext) -> str | None:
        if not self._turn_template:
            return None
        return self._render(self._turn_template, ctx)

    def turn_template_source(self) -> str | None:
        return self._turn_template or None

    def turn_format_kwargs(self, ctx: PromptContext) -> dict[str, object]:
        _ = ctx
        return {}

    def _render(self, template: str, ctx: PromptContext) -> str:
        _ = ctx
        return template.strip()


class BudgetPrompt(_BaseRolloutPrompt):
    """Announce the turn budget and report remaining turns each turn."""

    def __init__(self) -> None:
        super().__init__(
            user_files=("default/budget_user.md",),
            turn_files=("default/budget_turn.md",),
        )

    def user(self, ctx: PromptContext) -> str | None:
        if ctx.rollout_max_turns is None:
            return None
        return super().user(ctx)

    def turn(self, ctx: PromptContext) -> str | None:
        if ctx.turns_remaining is None or ctx.rollout_max_turns is None:
            return None
        return super().turn(ctx)

    def turn_format_kwargs(self, ctx: PromptContext) -> dict[str, object]:
        if ctx.rollout_max_turns is None:
            return {}
        return {"rollout_max_turns": ctx.rollout_max_turns}

    def _render(self, template: str, ctx: PromptContext) -> str:
        return template.format(
            turns_remaining=ctx.turns_remaining,
            **self.turn_format_kwargs(ctx),
        ).strip()


@dataclass(frozen=True)
class StaticRolloutPrompt:
    """Simple fixed prompt used by tests to exercise all hook positions."""

    system_text: str | None = None
    user_text: str | None = None
    turn_text: str | None = None

    def system(self, ctx: PromptContext) -> str | None:
        _ = ctx
        return self.system_text

    def user(self, ctx: PromptContext) -> str | None:
        _ = ctx
        return self.user_text

    def turn(self, ctx: PromptContext) -> str | None:
        _ = ctx
        return self.turn_text

    def turn_template_source(self) -> str | None:
        return self.turn_text

    def turn_format_kwargs(self, ctx: PromptContext) -> dict[str, object]:
        _ = ctx
        return {}
