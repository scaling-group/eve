"""Tests for the low-cost OpenRouter GPT cache canary."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

from scaling_evolve.providers.agent.openrouter_cache_canary import (
    CanaryAssessment,
    assess_cache_growth,
    build_canary_candidate,
    build_canary_instruction,
    collect_generation_ids_from_stdout,
    fetch_canary_generations,
)
from scaling_evolve.providers.agent.openrouter_cache_report import GenerationUsage


def test_assess_cache_growth_fails_when_cached_tokens_plateau_at_system_only() -> None:
    generations = [
        GenerationUsage(
            generation_id="gen-1",
            native_tokens_prompt=5000,
            native_tokens_cached=0,
            cache_discount=0.0,
            finish_reason="tool_calls",
        ),
        GenerationUsage(
            generation_id="gen-2",
            native_tokens_prompt=7000,
            native_tokens_cached=3072,
            cache_discount=0.0,
            finish_reason="tool_calls",
        ),
        GenerationUsage(
            generation_id="gen-3",
            native_tokens_prompt=9000,
            native_tokens_cached=3072,
            cache_discount=0.0,
            finish_reason="stop",
        ),
    ]

    assessment = assess_cache_growth(generations, min_history_growth=256)

    assert assessment == CanaryAssessment(
        passed=False,
        baseline_cached_tokens=3072,
        max_cached_tokens=3072,
        history_growth_tokens=0,
        reason="conversation_history_cache_did_not_grow",
    )


def test_assess_cache_growth_passes_when_history_cache_grows_past_baseline() -> None:
    generations = [
        GenerationUsage(
            generation_id="gen-1",
            native_tokens_prompt=5000,
            native_tokens_cached=0,
            cache_discount=0.0,
            finish_reason="tool_calls",
        ),
        GenerationUsage(
            generation_id="gen-2",
            native_tokens_prompt=8000,
            native_tokens_cached=3072,
            cache_discount=0.0,
            finish_reason="tool_calls",
        ),
        GenerationUsage(
            generation_id="gen-3",
            native_tokens_prompt=10000,
            native_tokens_cached=3840,
            cache_discount=0.0,
            finish_reason="stop",
        ),
    ]

    assessment = assess_cache_growth(generations, min_history_growth=256)

    assert assessment == CanaryAssessment(
        passed=True,
        baseline_cached_tokens=3072,
        max_cached_tokens=3840,
        history_growth_tokens=768,
        reason="conversation_history_cache_grew",
    )


def test_collect_generation_ids_from_stdout_deduplicates_assistant_ids() -> None:
    stdout_text = "\n".join(
        [
            json.dumps({"type": "assistant", "message": {"id": "gen-1", "content": []}}),
            json.dumps({"type": "assistant", "message": {"id": "gen-1", "content": []}}),
            json.dumps({"type": "assistant", "message": {"id": "gen-2", "content": []}}),
            json.dumps({"type": "result", "subtype": "success"}),
        ]
    )

    assert collect_generation_ids_from_stdout(stdout_text) == ("gen-1", "gen-2")


def test_build_canary_candidate_creates_large_readable_python_file(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.py"

    build_canary_candidate(candidate_path)

    text = candidate_path.read_text(encoding="utf-8")
    assert "def compute_value" in text
    assert text.count("COEFFICIENT_") >= 128
    assert len(text.splitlines()) >= 140


def test_build_canary_instruction_mentions_candidate_and_single_numeric_edit() -> None:
    instruction = build_canary_instruction()

    assert "./candidate.py" in instruction
    assert "change exactly one numeric literal" in instruction
    assert "Do not do anything else" in instruction


def test_fetch_canary_generations_uses_extended_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, float]] = []

    def _fake_fetch(
        generation_id: str,
        *,
        api_key: str,
        max_attempts: int = 0,
        retry_delay_seconds: float = 0.0,
        base_url: str = "",
    ) -> GenerationUsage:
        _ = (api_key, base_url)
        calls.append((generation_id, max_attempts, retry_delay_seconds))
        return GenerationUsage(
            generation_id=generation_id,
            native_tokens_prompt=2000,
            native_tokens_cached=4096,
            cache_discount=0.1,
            finish_reason="stop",
        )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.openrouter_cache_canary.fetch_generation_usage",
        _fake_fetch,
    )

    generations, missing_ids = fetch_canary_generations(("gen-1", "gen-2"), api_key="token")

    assert [generation.generation_id for generation in generations] == ["gen-1", "gen-2"]
    assert missing_ids == ()
    assert calls == [
        ("gen-1", 6, 3.0),
        ("gen-2", 6, 3.0),
    ]


def test_fetch_canary_generations_skips_persistent_404s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_fetch(
        generation_id: str,
        *,
        api_key: str,
        max_attempts: int = 0,
        retry_delay_seconds: float = 0.0,
        base_url: str = "",
    ) -> GenerationUsage:
        _ = (api_key, max_attempts, retry_delay_seconds, base_url)
        if generation_id == "gen-missing":
            raise HTTPError(
                url="https://openrouter.ai/api/v1/generation?id=gen-missing",
                code=404,
                msg="Not Found",
                hdrs=None,
                fp=BytesIO(b""),
            )
        return GenerationUsage(
            generation_id=generation_id,
            native_tokens_prompt=2000,
            native_tokens_cached=4096,
            cache_discount=0.1,
            finish_reason="stop",
        )

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.openrouter_cache_canary.fetch_generation_usage",
        _fake_fetch,
    )

    generations, missing_ids = fetch_canary_generations(
        ("gen-ok", "gen-missing"),
        api_key="token",
    )

    assert [generation.generation_id for generation in generations] == ["gen-ok"]
    assert missing_ids == ("gen-missing",)
