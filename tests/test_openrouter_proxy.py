"""Tests for OpenRouter proxy request normalization."""

from __future__ import annotations

import json
from collections.abc import Iterator

from scaling_evolve.providers.agent.drivers.openrouter_proxy import (
    normalize_openrouter_messages_request,
    normalize_openrouter_messages_response,
)


def _has_cache_control(value: object) -> bool:
    if isinstance(value, dict):
        if "cache_control" in value:
            return True
        return any(_has_cache_control(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_cache_control(item) for item in value)
    return False


def _iter_sse_payloads(body: bytes) -> Iterator[dict[str, object]]:
    for chunk in body.decode("utf-8").strip().split("\n\n"):
        lines = chunk.splitlines()
        data_lines = [line[5:].lstrip() for line in lines if line.startswith("data:")]
        if not data_lines:
            continue
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _collect_sse_tool_input(body: bytes, *, index: int) -> dict[str, object]:
    partial_json = ""
    for payload in _iter_sse_payloads(body):
        if payload.get("type") == "content_block_delta" and payload.get("index") == index:
            delta = payload.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "input_json_delta":
                partial_json += str(delta.get("partial_json", ""))
    return json.loads(partial_json)


def test_normalize_openrouter_messages_request_strips_cache_control_for_openai() -> None:
    payload = {
        "model": "openai/gpt-5.4-mini",
        "metadata": {
            "user_id": json.dumps(
                {
                    "device_id": "device-1",
                    "account_uuid": "",
                    "session_id": "session-123",
                }
            )
        },
        "system": [
            {
                "type": "text",
                "text": "x-anthropic-billing-header: cc_version=2.1.87; cch=session-123;",
            },
            {
                "type": "text",
                "text": "You are a Claude agent.",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Open ./candidate.py first."},
                    {
                        "type": "text",
                        "text": "Make one change.",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "value = 1",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ],
    }

    normalized = normalize_openrouter_messages_request(
        payload,
        pinned_provider="OpenAI",
    )

    assert normalized["provider"] == {
        "order": ["OpenAI"],
        "allow_fallbacks": False,
    }
    assert "cch=normalized" in normalized["system"][0]["text"]
    assert json.loads(normalized["metadata"]["user_id"])["session_id"] == "normalized-session"
    assert not _has_cache_control(normalized)


def test_normalize_openrouter_messages_request_keeps_cache_control_for_anthropic() -> None:
    payload = {
        "model": "anthropic/claude-sonnet-4",
        "metadata": {
            "user_id": json.dumps(
                {
                    "device_id": "device-1",
                    "account_uuid": "",
                    "session_id": "session-123",
                }
            )
        },
        "system": [
            {
                "type": "text",
                "text": "x-anthropic-billing-header: cc_version=2.1.87; cch=session-123;",
            },
            {
                "type": "text",
                "text": "You are Claude Code.",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Continue.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    }

    normalized = normalize_openrouter_messages_request(
        payload,
        pinned_provider="Anthropic",
    )

    assert normalized["provider"] == {
        "order": ["Anthropic"],
        "allow_fallbacks": False,
    }
    assert _has_cache_control(normalized)


def test_normalize_openrouter_messages_response_strips_empty_pages_from_json_tool_use() -> None:
    payload = {
        "id": "msg-1",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "call-1",
                "name": "Read",
                "input": {
                    "file_path": "./candidate.py",
                    "offset": 1,
                    "limit": 400,
                    "pages": "",
                },
            }
        ],
    }

    normalized = json.loads(
        normalize_openrouter_messages_response(
            json.dumps(payload).encode("utf-8"),
            content_type="application/json",
        ).decode("utf-8")
    )

    assert normalized["content"][0]["input"] == {
        "file_path": "./candidate.py",
        "offset": 1,
        "limit": 400,
    }


def test_normalize_openrouter_messages_response_strips_empty_pages_from_streaming_tool_use() -> (
    None
):
    body = "\n\n".join(
        [
            "event: message_start\ndata: "
            + json.dumps(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg-1",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                    },
                },
                separators=(",", ":"),
            ),
            "event: content_block_start\ndata: "
            + json.dumps(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "Read",
                        "input": {},
                    },
                },
                separators=(",", ":"),
            ),
            "event: content_block_delta\ndata: "
            + json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(
                            {
                                "file_path": "./candidate.py",
                                "offset": 1,
                                "limit": 400,
                                "pages": "",
                            },
                            separators=(",", ":"),
                        ),
                    },
                },
                separators=(",", ":"),
            ),
            "event: content_block_stop\ndata: "
            + json.dumps(
                {"type": "content_block_stop", "index": 0},
                separators=(",", ":"),
            ),
            "event: message_stop\ndata: "
            + json.dumps({"type": "message_stop"}, separators=(",", ":")),
            "event: data\ndata: [DONE]",
        ]
    ).encode("utf-8")

    normalized = normalize_openrouter_messages_response(
        body,
        content_type="text/event-stream",
    )

    assert _collect_sse_tool_input(normalized, index=0) == {
        "file_path": "./candidate.py",
        "offset": 1,
        "limit": 400,
    }


def test_normalize_openrouter_messages_response_keeps_valid_pages_in_streaming_tool_use() -> None:
    body = "\n\n".join(
        [
            "event: content_block_start\ndata: "
            + json.dumps(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "Read",
                        "input": {},
                    },
                },
                separators=(",", ":"),
            ),
            "event: content_block_delta\ndata: "
            + json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(
                            {
                                "file_path": "./candidate.py",
                                "offset": 1,
                                "limit": 400,
                                "pages": "1",
                            },
                            separators=(",", ":"),
                        ),
                    },
                },
                separators=(",", ":"),
            ),
            "event: content_block_stop\ndata: "
            + json.dumps(
                {"type": "content_block_stop", "index": 0},
                separators=(",", ":"),
            ),
        ]
    ).encode("utf-8")

    normalized = normalize_openrouter_messages_response(
        body,
        content_type="text/event-stream",
    )

    assert _collect_sse_tool_input(normalized, index=0) == {
        "file_path": "./candidate.py",
        "offset": 1,
        "limit": 400,
        "pages": "1",
    }


def test_normalize_openrouter_messages_response_returns_original_body_for_malformed_json() -> None:
    body = b"{not-json}"

    normalized = normalize_openrouter_messages_response(
        body,
        content_type="application/json",
    )

    assert normalized == body


def test_normalize_openrouter_messages_response_returns_original_body_for_malformed_stream() -> (
    None
):
    body = b"event: content_block_delta\ndata: {not-json}\n\n"

    normalized = normalize_openrouter_messages_response(
        body,
        content_type="text/event-stream",
    )

    assert normalized == body
