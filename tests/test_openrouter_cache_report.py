"""Tests for OpenRouter generation-endpoint cache reporting."""

from __future__ import annotations

import json
import sqlite3
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

from scaling_evolve.providers.agent.openrouter_cache_report import (
    EdgeCacheInput,
    GenerationUsage,
    build_run_cache_report,
    collect_edge_cache_inputs,
    fetch_generation_usage,
    provider_usage_from_generations,
)


def test_collect_edge_cache_inputs_reads_response_artifacts_and_execution_mode(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-123"
    artifacts_root = run_root / "artifacts"
    responses_root = artifacts_root / "responses"
    state_root = artifacts_root / "state"
    responses_root.mkdir(parents=True)
    state_root.mkdir(parents=True)

    response_path = responses_root / "abcd__edge-0001.claude_code.stdout.jsonl"
    response_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "assistant", "message": {"id": "gen-1", "content": []}}),
                json.dumps({"type": "assistant", "message": {"id": "gen-2", "content": []}}),
                json.dumps({"type": "result", "subtype": "success"}),
            ]
        ),
        encoding="utf-8",
    )

    mutation_result_path = state_root / "result__edge-0001.mutation_result.json"
    mutation_result_path.write_text(
        json.dumps(
            {
                "artifact_refs": [
                    {
                        "kind": "model.response.raw.json",
                        "uri": str(response_path),
                    }
                ],
                "child_runtime_state": {
                    "metadata": {
                        "actual_execution_mode": "fork",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    sqlite_path = run_root / "lineage.sqlite3"
    conn = sqlite3.connect(sqlite_path)
    conn.execute(
        """
        CREATE TABLE edges (
          run_id TEXT NOT NULL,
          edge_id TEXT NOT NULL,
          result_ref TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE edge_executions (
          run_id TEXT NOT NULL,
          edge_id TEXT NOT NULL,
          input_tokens INTEGER NOT NULL,
          output_tokens INTEGER NOT NULL,
          cache_read_input_tokens INTEGER NOT NULL,
          cache_creation_input_tokens INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO edges(run_id, edge_id, result_ref) VALUES (?, ?, ?)",
        (
            "run-123",
            "edge-0001",
            json.dumps({"uri": str(mutation_result_path)}),
        ),
    )
    conn.execute(
        """
        INSERT INTO edge_executions(
          run_id,
          edge_id,
          input_tokens,
          output_tokens,
          cache_read_input_tokens,
          cache_creation_input_tokens
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("run-123", "edge-0001", 1000, 50, 300, 0),
    )
    conn.commit()
    conn.close()

    inputs = collect_edge_cache_inputs(run_root)

    assert inputs == [
        EdgeCacheInput(
            edge_id="edge-0001",
            execution_mode="fork",
            response_path=response_path,
            generation_ids=("gen-1", "gen-2"),
            stream_input_tokens=1000,
            stream_output_tokens=50,
            stream_cache_read_tokens=300,
            stream_cache_creation_tokens=0,
        )
    ]


def test_build_run_cache_report_aggregates_generation_endpoint_metrics() -> None:
    inputs = [
        EdgeCacheInput(
            edge_id="edge-0001",
            execution_mode="spawn",
            response_path=Path("/tmp/edge-0001.jsonl"),
            generation_ids=("gen-1", "gen-2"),
            stream_input_tokens=1000,
            stream_output_tokens=50,
            stream_cache_read_tokens=100,
            stream_cache_creation_tokens=0,
        ),
        EdgeCacheInput(
            edge_id="edge-0002",
            execution_mode="fork",
            response_path=Path("/tmp/edge-0002.jsonl"),
            generation_ids=("gen-3",),
            stream_input_tokens=800,
            stream_output_tokens=40,
            stream_cache_read_tokens=50,
            stream_cache_creation_tokens=0,
        ),
    ]
    generations = {
        "gen-1": GenerationUsage(
            generation_id="gen-1",
            native_tokens_prompt=500,
            native_tokens_cached=0,
            cache_discount=0.0,
            finish_reason="tool_calls",
        ),
        "gen-2": GenerationUsage(
            generation_id="gen-2",
            native_tokens_prompt=700,
            native_tokens_cached=400,
            cache_discount=0.25,
            finish_reason="stop",
        ),
        "gen-3": GenerationUsage(
            generation_id="gen-3",
            native_tokens_prompt=600,
            native_tokens_cached=300,
            cache_discount=0.10,
            finish_reason="stop",
        ),
    }

    report = build_run_cache_report(inputs, fetch_generation=generations.__getitem__)

    assert report.overall.edge_count == 2
    assert report.overall.request_count == 3
    assert report.overall.stream_cache_read_tokens == 150
    assert report.overall.generation_cached_tokens == 700
    assert report.overall.cache_discount == 0.35
    assert report.overall.generation_hit_rate == 700 / 1800

    assert report.by_mode["spawn"].generation_cached_tokens == 400
    assert report.by_mode["fork"].generation_cached_tokens == 300
    assert report.edges[0].generation_cached_tokens == 400
    assert report.edges[1].generation_cached_tokens == 300


def test_fetch_generation_usage_retries_transient_404(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        def read(self) -> bytes:
            return json.dumps(
                {
                    "data": {
                        "native_tokens_prompt": 1234,
                        "native_tokens_cached": 512,
                        "cache_discount": 0.42,
                        "total_cost": 0.77,
                        "finish_reason": "stop",
                    }
                }
            ).encode("utf-8")

    def _fake_urlopen(req, timeout=300):  # noqa: ANN001
        _ = (req, timeout)
        calls["count"] += 1
        if calls["count"] == 1:
            raise HTTPError(
                url="https://openrouter.ai/api/v1/generation?id=gen-123",
                code=404,
                msg="Not Found",
                hdrs=None,
                fp=BytesIO(b""),
            )
        return _Response()

    monkeypatch.setattr(
        "scaling_evolve.providers.agent.openrouter_cache_report.request.urlopen",
        _fake_urlopen,
    )

    usage = fetch_generation_usage("gen-123", api_key="token")

    assert calls["count"] == 2
    assert usage == GenerationUsage(
        generation_id="gen-123",
        native_tokens_prompt=1234,
        native_tokens_cached=512,
        cache_discount=0.42,
        finish_reason="stop",
        total_cost=0.77,
    )


def test_provider_usage_from_generations_builds_canonical_budget_usage() -> None:
    usage = provider_usage_from_generations(
        [
            GenerationUsage(
                generation_id="gen-1",
                native_tokens_prompt=4000,
                native_tokens_cached=1024,
                cache_discount=0.1,
                finish_reason="tool_calls",
                native_tokens_completion=50,
                total_cost=0.2,
            ),
            GenerationUsage(
                generation_id="gen-2",
                native_tokens_prompt=5000,
                native_tokens_cached=2048,
                cache_discount=0.2,
                finish_reason="stop",
                native_tokens_completion=70,
                total_cost=0.3,
            ),
        ],
        wallclock_seconds=4.5,
        agent_turns=3,
    )

    assert usage.input_tokens == 5928
    assert usage.output_tokens == 120
    assert usage.cache_read_tokens == 3072
    assert usage.model_cost_usd == 0.5
    assert usage.wallclock_seconds == 4.5
    assert usage.agent_turns == 3
