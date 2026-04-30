"""Helpers for OpenRouter generation-endpoint cache reporting."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast
from urllib import error, request

from scaling_evolve.core.mutation import ProviderUsage


@dataclass(frozen=True)
class EdgeCacheInput:
    edge_id: str
    execution_mode: str
    response_path: Path
    generation_ids: tuple[str, ...]
    stream_input_tokens: int
    stream_output_tokens: int
    stream_cache_read_tokens: int
    stream_cache_creation_tokens: int


@dataclass(frozen=True)
class GenerationUsage:
    generation_id: str
    native_tokens_prompt: int
    native_tokens_cached: int
    cache_discount: float
    finish_reason: str | None
    native_tokens_completion: int = 0
    total_cost: float = 0.0


@dataclass(frozen=True)
class CacheSummary:
    edge_count: int
    request_count: int
    stream_input_tokens: int
    stream_output_tokens: int
    stream_cache_read_tokens: int
    stream_cache_creation_tokens: int
    generation_prompt_tokens: int
    generation_cached_tokens: int
    cache_discount: float
    total_cost: float = 0.0

    @property
    def generation_hit_rate(self) -> float:
        if self.generation_prompt_tokens <= 0:
            return 0.0
        return self.generation_cached_tokens / self.generation_prompt_tokens

    @property
    def stream_hit_rate(self) -> float:
        if self.stream_input_tokens <= 0:
            return 0.0
        return self.stream_cache_read_tokens / self.stream_input_tokens

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["generation_hit_rate"] = self.generation_hit_rate
        payload["stream_hit_rate"] = self.stream_hit_rate
        return payload


@dataclass(frozen=True)
class EdgeCacheReport:
    edge_id: str
    execution_mode: str
    request_count: int
    stream_input_tokens: int
    stream_output_tokens: int
    stream_cache_read_tokens: int
    stream_cache_creation_tokens: int
    generation_prompt_tokens: int
    generation_cached_tokens: int
    cache_discount: float
    generation_ids: tuple[str, ...]
    total_cost: float = 0.0

    @property
    def generation_hit_rate(self) -> float:
        if self.generation_prompt_tokens <= 0:
            return 0.0
        return self.generation_cached_tokens / self.generation_prompt_tokens

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["generation_hit_rate"] = self.generation_hit_rate
        return payload


@dataclass(frozen=True)
class RunCacheReport:
    run_root: Path
    edges: tuple[EdgeCacheReport, ...]
    by_mode: dict[str, CacheSummary]
    overall: CacheSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "run_root": str(self.run_root),
            "edges": [edge.to_dict() for edge in self.edges],
            "by_mode": {mode: summary.to_dict() for mode, summary in self.by_mode.items()},
            "overall": self.overall.to_dict(),
        }


def collect_edge_cache_inputs(run_root: Path) -> list[EdgeCacheInput]:
    sqlite_path = run_root / "lineage.sqlite3"
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
              edges.edge_id,
              edges.result_ref,
              COALESCE(edge_executions.input_tokens, 0) AS input_tokens,
              COALESCE(edge_executions.output_tokens, 0) AS output_tokens,
              COALESCE(edge_executions.cache_read_input_tokens, 0) AS cache_read_input_tokens,
              COALESCE(edge_executions.cache_creation_input_tokens, 0)
                AS cache_creation_input_tokens
            FROM edges
            LEFT JOIN edge_executions
              ON edges.run_id = edge_executions.run_id
             AND edges.edge_id = edge_executions.edge_id
            ORDER BY edges.edge_id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    collected: list[EdgeCacheInput] = []
    for row in rows:
        result_ref = _json_object(row["result_ref"])
        mutation_result_path = _path_from_uri(result_ref.get("uri"))
        if mutation_result_path is None:
            continue
        mutation_result = _json_object(_read_json(mutation_result_path))
        response_path = _response_path_from_mutation_result(mutation_result)
        if response_path is None or not response_path.exists():
            continue
        generation_ids = extract_generation_ids(response_path)
        if not generation_ids:
            continue
        collected.append(
            EdgeCacheInput(
                edge_id=str(row["edge_id"]),
                execution_mode=_execution_mode_from_mutation_result(mutation_result),
                response_path=response_path,
                generation_ids=generation_ids,
                stream_input_tokens=int(row["input_tokens"]),
                stream_output_tokens=int(row["output_tokens"]),
                stream_cache_read_tokens=int(row["cache_read_input_tokens"]),
                stream_cache_creation_tokens=int(row["cache_creation_input_tokens"]),
            )
        )
    return collected


def extract_generation_ids(response_path: Path) -> tuple[str, ...]:
    seen: set[str] = set()
    generation_ids: list[str] = []
    for raw_line in response_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") != "assistant":
            continue
        message = payload.get("message")
        if not isinstance(message, dict):
            continue
        generation_id = message.get("id")
        if not isinstance(generation_id, str) or not generation_id.startswith("gen-"):
            continue
        if generation_id in seen:
            continue
        seen.add(generation_id)
        generation_ids.append(generation_id)
    return tuple(generation_ids)


def fetch_generation_usage(
    generation_id: str,
    *,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1/generation",
    max_attempts: int = 3,
    retry_delay_seconds: float = 2.0,
) -> GenerationUsage:
    endpoint = f"{base_url}?id={generation_id}"
    req = request.Request(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    payload: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with request.urlopen(req, timeout=300) as response:
                payload = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
            break
        except error.HTTPError as exc:
            if exc.code != 404 or attempt >= max_attempts:
                raise
            time.sleep(retry_delay_seconds)
    if payload is None:
        raise RuntimeError(f"Failed to fetch generation payload for `{generation_id}`.")
    data = _json_object(payload.get("data"))
    return GenerationUsage(
        generation_id=generation_id,
        native_tokens_prompt=int(data.get("native_tokens_prompt") or 0),
        native_tokens_cached=int(data.get("native_tokens_cached") or 0),
        native_tokens_completion=int(data.get("native_tokens_completion") or 0),
        cache_discount=float(data.get("cache_discount") or 0.0),
        total_cost=float(data.get("total_cost") or 0.0),
        finish_reason=(
            str(data.get("finish_reason")) if data.get("finish_reason") is not None else None
        ),
    )


def build_run_cache_report(
    edge_inputs: Sequence[EdgeCacheInput],
    *,
    fetch_generation: Callable[[str], GenerationUsage],
    run_root: Path | None = None,
) -> RunCacheReport:
    edge_reports: list[EdgeCacheReport] = []
    buckets: dict[str, list[EdgeCacheReport]] = defaultdict(list)
    for edge in edge_inputs:
        generations = [fetch_generation(generation_id) for generation_id in edge.generation_ids]
        edge_report = EdgeCacheReport(
            edge_id=edge.edge_id,
            execution_mode=edge.execution_mode,
            request_count=len(generations),
            stream_input_tokens=edge.stream_input_tokens,
            stream_output_tokens=edge.stream_output_tokens,
            stream_cache_read_tokens=edge.stream_cache_read_tokens,
            stream_cache_creation_tokens=edge.stream_cache_creation_tokens,
            generation_prompt_tokens=sum(item.native_tokens_prompt for item in generations),
            generation_cached_tokens=sum(item.native_tokens_cached for item in generations),
            cache_discount=sum(item.cache_discount for item in generations),
            total_cost=sum(item.total_cost for item in generations),
            generation_ids=edge.generation_ids,
        )
        edge_reports.append(edge_report)
        buckets[edge.execution_mode].append(edge_report)

    overall = _summarize_reports(edge_reports)
    by_mode = {mode: _summarize_reports(reports) for mode, reports in buckets.items()}
    return RunCacheReport(
        run_root=run_root or Path("."),
        edges=tuple(edge_reports),
        by_mode=by_mode,
        overall=overall,
    )


def report_for_run_root(
    run_root: Path,
    *,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1/generation",
) -> RunCacheReport:
    inputs = collect_edge_cache_inputs(run_root)
    return build_run_cache_report(
        inputs,
        fetch_generation=lambda generation_id: fetch_generation_usage(
            generation_id,
            api_key=api_key,
            base_url=base_url,
        ),
        run_root=run_root,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run",
        help="Run root path or run id to search under .runs/",
    )
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1/generation",
        help="Override the OpenRouter generation endpoint.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the JSON report.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing required environment variable `OPENROUTER_API_KEY`.")
    run_root = resolve_run_root(Path.cwd(), args.run)
    report = report_for_run_root(run_root, api_key=api_key, base_url=args.base_url)
    rendered = json.dumps(report.to_dict(), indent=2)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def resolve_run_root(cwd: Path, run: str) -> Path:
    candidate = Path(run)
    if candidate.exists():
        return candidate.resolve()
    matches = sorted(cwd.glob(f".runs/**/{run}"))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(f"Could not resolve run root for `{run}`.")
    raise ValueError(f"Run id `{run}` matched multiple run roots; pass an explicit path.")


def provider_usage_from_generations(
    generations: Sequence[GenerationUsage],
    *,
    wallclock_seconds: float = 0.0,
    agent_turns: int = 0,
) -> ProviderUsage:
    total_prompt_tokens = sum(item.native_tokens_prompt for item in generations)
    total_cached_tokens = sum(item.native_tokens_cached for item in generations)
    total_completion_tokens = sum(item.native_tokens_completion for item in generations)
    return ProviderUsage(
        input_tokens=max(total_prompt_tokens - total_cached_tokens, 0),
        output_tokens=total_completion_tokens,
        cache_read_tokens=total_cached_tokens,
        cache_creation_tokens=0,
        model_cost_usd=sum(item.total_cost for item in generations),
        wallclock_seconds=wallclock_seconds,
        agent_turns=agent_turns,
    )


def _summarize_reports(reports: Iterable[EdgeCacheReport]) -> CacheSummary:
    report_list = tuple(reports)
    return CacheSummary(
        edge_count=len(report_list),
        request_count=sum(report.request_count for report in report_list),
        stream_input_tokens=sum(report.stream_input_tokens for report in report_list),
        stream_output_tokens=sum(report.stream_output_tokens for report in report_list),
        stream_cache_read_tokens=sum(report.stream_cache_read_tokens for report in report_list),
        stream_cache_creation_tokens=sum(
            report.stream_cache_creation_tokens for report in report_list
        ),
        generation_prompt_tokens=sum(report.generation_prompt_tokens for report in report_list),
        generation_cached_tokens=sum(report.generation_cached_tokens for report in report_list),
        cache_discount=sum(report.cache_discount for report in report_list),
        total_cost=sum(report.total_cost for report in report_list),
    )


def _execution_mode_from_mutation_result(payload: dict[str, Any]) -> str:
    child_runtime_state = payload.get("child_runtime_state")
    if not isinstance(child_runtime_state, dict):
        return "unknown"
    metadata = child_runtime_state.get("metadata")
    if not isinstance(metadata, dict):
        return "unknown"
    mode = metadata.get("actual_execution_mode")
    return str(mode) if isinstance(mode, str) and mode else "unknown"


def _response_path_from_mutation_result(payload: dict[str, Any]) -> Path | None:
    artifact_refs = payload.get("artifact_refs")
    if not isinstance(artifact_refs, list):
        return None
    for artifact in artifact_refs:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("kind") != "model.response.raw.json":
            continue
        return _path_from_uri(artifact.get("uri"))
    return None


def _path_from_uri(value: object) -> Path | None:
    if not isinstance(value, str) or value == "":
        return None
    return Path(value)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
    raise ValueError("Expected a JSON object payload.")


if __name__ == "__main__":
    raise SystemExit(main())
