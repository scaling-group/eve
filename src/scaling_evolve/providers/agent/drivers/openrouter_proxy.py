"""OpenRouter proxy for pinning provider routing and normalizing cache keys."""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib import error, request

_CCH_RE = re.compile(r"cch=[^;\s]+")


# Compatibility hack 1/2:
# WHY: Claude Code injects Anthropic `cache_control` markers into request blocks. Those
# markers are fine on Anthropic-native paths, but on the `Claude Code -> OpenRouter -> GPT`
# path they can destabilize the converted prompt prefix and collapse GPT cache reuse.
# WHEN TO REMOVE: when a fresh canary and corrected generation report stay healthy without
# stripping `cache_control` on the current Claude Code + OpenRouter GPT stack.
# HOW TO VERIFY: run `scripts/experiments/check_openrouter_cache_canary.py` and
# `scripts/experiments/report_openrouter_generation_cache.py <run-id>` after upgrades.
def _strip_cache_control(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: cast(Any, _strip_cache_control(item))
            for key, item in value.items()
            if key != "cache_control"
        }
    if isinstance(value, list):
        return [cast(Any, _strip_cache_control(item)) for item in value]
    return value


# Compatibility hack 2/2:
# WHY: some Claude Code + GPT Read tool calls arrive as `pages: ""`, which Claude Code
# rejects before the Read executes. Removing only the empty-string variant preserves valid
# `pages` values like `"1"` while eliminating the wasted error/retry turn.
# WHEN TO REMOVE: when fresh probe/smoke runs on the current Claude Code + OpenRouter GPT
# stack stop emitting empty `pages` in saved Read tool calls and still show zero
# `invalid_pages` errors without proxy rewriting.
# HOW TO VERIFY: run the probe/smoke configs, grep run artifacts for `invalid_pages`,
# and inspect the first Read tool_use blocks in `*.claude_code.stdout.jsonl`.
def _strip_invalid_read_pages(value: object) -> object:
    if isinstance(value, dict):
        normalized = {
            key: cast(Any, _strip_invalid_read_pages(item)) for key, item in value.items()
        }
        if normalized.get("type") == "tool_use" and normalized.get("name") == "Read":
            tool_input = normalized.get("input")
            if isinstance(tool_input, dict) and tool_input.get("pages") == "":
                normalized["input"] = {
                    key: item for key, item in tool_input.items() if key != "pages"
                }
        return normalized
    if isinstance(value, list):
        return [cast(Any, _strip_invalid_read_pages(item)) for item in value]
    return value


def _should_strip_cache_control(payload: dict[str, Any], *, pinned_provider: str) -> bool:
    model = payload.get("model")
    if isinstance(model, str) and model.startswith("openai/"):
        return True
    return pinned_provider.lower() == "openai"


def normalize_openrouter_messages_request(
    payload: dict[str, Any],
    *,
    pinned_provider: str,
) -> dict[str, Any]:
    """Inject OpenRouter provider routing and normalize known cache-busting fields."""

    normalized = dict(payload)
    normalized["provider"] = {
        "order": [pinned_provider],
        "allow_fallbacks": False,
    }

    metadata_raw = normalized.get("metadata")
    if isinstance(metadata_raw, dict):
        metadata = cast(dict[str, Any], metadata_raw)
        user_id = metadata.get("user_id")
        if isinstance(user_id, str):
            try:
                user_payload = json.loads(user_id)
            except json.JSONDecodeError:
                user_payload = None
            if isinstance(user_payload, dict):
                user_payload["session_id"] = "normalized-session"
                metadata["user_id"] = json.dumps(user_payload, separators=(",", ":"))

    system_raw = normalized.get("system")
    if isinstance(system_raw, list) and system_raw:
        system = cast(list[Any], system_raw)
        first_raw = system[0]
        if isinstance(first_raw, dict):
            first = cast(dict[str, Any], first_raw)
            text = first.get("text")
            if isinstance(text, str):
                first["text"] = _CCH_RE.sub("cch=normalized", text)

    if _should_strip_cache_control(normalized, pinned_provider=pinned_provider):
        normalized = cast(dict[str, Any], _strip_cache_control(normalized))

    return normalized


def _parse_sse_event(chunk: str) -> tuple[str | None, str | None, dict[str, Any] | None]:
    event_name: str | None = None
    data_lines: list[str] = []
    for line in chunk.splitlines():
        if line.startswith("event:"):
            event_name = line.partition(":")[2].lstrip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.partition(":")[2].lstrip())
    if not data_lines:
        return (event_name, None, None)
    data_text = "\n".join(data_lines)
    try:
        payload = json.loads(data_text)
    except json.JSONDecodeError:
        return (event_name, data_text, None)
    if isinstance(payload, dict):
        return (event_name, data_text, cast(dict[str, Any], payload))
    return (event_name, data_text, None)


def _serialize_sse_event(
    *,
    event_name: str | None,
    data_text: str | None = None,
    payload: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    if event_name is not None:
        lines.append(f"event: {event_name}")
    serialized = data_text
    if payload is not None:
        serialized = json.dumps(payload, separators=(",", ":"))
    if serialized is not None:
        for line in serialized.splitlines():
            lines.append(f"data: {line}")
    return "\n".join(lines)


# The response-side pages hack is intentionally conservative. If OpenRouter changes its SSE
# framing enough that we cannot confidently parse it, callers should fall back to the original
# body rather than silently rewriting an unknown format.
def _normalize_streaming_response(body: bytes) -> bytes:
    text = body.decode("utf-8")
    chunks = [chunk for chunk in text.replace("\r\n", "\n").split("\n\n") if chunk.strip() != ""]
    normalized_chunks: list[str] = []
    active_tool_uses: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
        event_name, data_text, payload = _parse_sse_event(chunk)
        if payload is None:
            normalized_chunks.append(
                _serialize_sse_event(event_name=event_name, data_text=data_text)
            )
            continue

        event_type = str(payload.get("type", ""))
        index_raw = payload.get("index")
        index = index_raw if isinstance(index_raw, int) else None

        if event_type == "content_block_start" and index is not None:
            content_block = payload.get("content_block")
            if (
                isinstance(content_block, dict)
                and content_block.get("type") == "tool_use"
                and content_block.get("name") == "Read"
            ):
                active_tool_uses[index] = {
                    "start": cast(dict[str, Any], _strip_invalid_read_pages(payload)),
                    "deltas": [],
                    "fragments": [],
                }
                continue

        if index is not None and index in active_tool_uses and event_type == "content_block_delta":
            delta = payload.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "input_json_delta":
                active_tool_uses[index]["deltas"].append(payload)
                active_tool_uses[index]["fragments"].append(str(delta.get("partial_json", "")))
                continue

        if index is not None and index in active_tool_uses and event_type == "content_block_stop":
            buffered = active_tool_uses.pop(index)
            normalized_chunks.append(
                _serialize_sse_event(
                    event_name="content_block_start",
                    payload=cast(dict[str, Any], buffered["start"]),
                )
            )
            partial_json = "".join(cast(list[str], buffered["fragments"]))
            try:
                tool_input = json.loads(partial_json) if partial_json else {}
            except json.JSONDecodeError:
                for delta_payload in cast(list[dict[str, Any]], buffered["deltas"]):
                    normalized_chunks.append(
                        _serialize_sse_event(
                            event_name="content_block_delta",
                            payload=delta_payload,
                        )
                    )
            else:
                normalized_input = _strip_invalid_read_pages(
                    {
                        "type": "tool_use",
                        "name": cast(dict[str, Any], buffered["start"])
                        .get("content_block", {})
                        .get("name"),
                        "input": tool_input,
                    }
                )
                normalized_tool_input = cast(dict[str, Any], normalized_input).get("input")
                if normalized_tool_input != tool_input:
                    normalized_chunks.append(
                        _serialize_sse_event(
                            event_name="content_block_delta",
                            payload={
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": json.dumps(
                                        normalized_tool_input,
                                        separators=(",", ":"),
                                    ),
                                },
                            },
                        )
                    )
                else:
                    for delta_payload in cast(list[dict[str, Any]], buffered["deltas"]):
                        normalized_chunks.append(
                            _serialize_sse_event(
                                event_name="content_block_delta",
                                payload=delta_payload,
                            )
                        )
            normalized_chunks.append(
                _serialize_sse_event(event_name="content_block_stop", payload=payload)
            )
            continue

        normalized_chunks.append(_serialize_sse_event(event_name=event_name, payload=payload))

    for buffered in active_tool_uses.values():
        normalized_chunks.append(
            _serialize_sse_event(
                event_name="content_block_start",
                payload=cast(dict[str, Any], buffered["start"]),
            )
        )
        for delta_payload in cast(list[dict[str, Any]], buffered["deltas"]):
            normalized_chunks.append(
                _serialize_sse_event(event_name="content_block_delta", payload=delta_payload)
            )

    if not normalized_chunks:
        return body
    return ("\n\n".join(normalized_chunks) + "\n\n").encode("utf-8")


def normalize_openrouter_messages_response(
    body: bytes,
    *,
    content_type: str,
) -> bytes:
    if "text/event-stream" in content_type:
        try:
            return _normalize_streaming_response(body)
        except (UnicodeDecodeError, ValueError, TypeError):
            return body
    if "application/json" in content_type:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body
        normalized = _strip_invalid_read_pages(payload)
        return json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return body


class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        server = self.server
        assert isinstance(server, _ProxyServer)

        content_length = int(self.headers.get("content-length", "0"))
        raw_body = self.rfile.read(content_length)
        payload = json.loads(raw_body.decode("utf-8"))
        normalized = normalize_openrouter_messages_request(
            payload,
            pinned_provider=server.pinned_provider,
        )
        encoded = json.dumps(normalized).encode("utf-8")
        forward = request.Request(
            server.target_base_url + self.path,
            data=encoded,
            headers={
                "Authorization": self.headers.get("Authorization", ""),
                "Content-Type": "application/json",
                "anthropic-version": self.headers.get("anthropic-version", "2023-06-01"),
            },
            method="POST",
        )

        try:
            with request.urlopen(forward, timeout=server.timeout_seconds) as response:
                body = normalize_openrouter_messages_response(
                    response.read(),
                    content_type=response.headers.get("Content-Type", "application/json"),
                )
                self.send_response(response.status)
                self.send_header(
                    "Content-Type",
                    response.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except error.HTTPError as http_error:
            body = normalize_openrouter_messages_response(
                http_error.read(),
                content_type=http_error.headers.get("Content-Type", "application/json"),
            )
            self.send_response(http_error.code)
            self.send_header(
                "Content-Type",
                http_error.headers.get("Content-Type", "application/json"),
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        _ = (format, args)


class _ProxyServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        target_base_url: str,
        pinned_provider: str,
        timeout_seconds: float,
    ) -> None:
        super().__init__(server_address, _ProxyHandler)
        self.target_base_url = target_base_url.rstrip("/")
        self.pinned_provider = pinned_provider
        self.timeout_seconds = timeout_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-base-url", required=True)
    parser.add_argument("--pinned-provider", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()

    server = _ProxyServer(
        ("127.0.0.1", args.listen_port),
        target_base_url=args.target_base_url,
        pinned_provider=args.pinned_provider,
        timeout_seconds=args.timeout_seconds,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
