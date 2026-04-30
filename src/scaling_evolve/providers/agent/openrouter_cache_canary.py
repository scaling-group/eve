"""Low-cost cache canary for the Claude Code -> OpenRouter GPT path."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib import error

from scaling_evolve.providers.agent.openrouter_cache_report import (
    GenerationUsage,
    fetch_generation_usage,
)


@dataclass(frozen=True)
class CanaryAssessment:
    passed: bool
    baseline_cached_tokens: int
    max_cached_tokens: int
    history_growth_tokens: int
    reason: str


@dataclass(frozen=True)
class CanaryResult:
    claude_code_version: str | None
    model: str
    generation_ids: tuple[str, ...]
    missing_generation_ids: tuple[str, ...]
    cached_sequence: tuple[int, ...]
    prompt_sequence: tuple[int, ...]
    assessment: CanaryAssessment
    stdout_path: str
    working_directory: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["assessment"] = asdict(self.assessment)
        return payload


def build_canary_candidate(path: Path, *, coefficient_count: int = 160) -> None:
    lines = [
        '"""OpenRouter GPT cache canary candidate."""',
        "",
        "from __future__ import annotations",
        "",
        "# Large stable file content keeps the read tool result well above cache thresholds.",
    ]
    for index in range(coefficient_count):
        lines.append(f"COEFFICIENT_{index:03d} = {index + 1}")
    lines.extend(
        [
            "",
            "def compute_value() -> int:",
            "    total = 0",
            f"    for index in range({coefficient_count}):",
            "        total += globals()[f'COEFFICIENT_{index:03d}']",
            "    return total",
            "",
            "def sentinel() -> int:",
            "    return COEFFICIENT_000",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_canary_instruction() -> str:
    return (
        "Open ./candidate.py, change exactly one numeric literal from 1 to 2, and save the file. "
        "Read the file before editing. Edit only ./candidate.py. Do not do anything else."
    )


def collect_generation_ids_from_stdout(stdout_text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    generation_ids: list[str] = []
    for raw_line in stdout_text.splitlines():
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


def assess_cache_growth(
    generations: list[GenerationUsage],
    *,
    min_history_growth: int = 256,
) -> CanaryAssessment:
    cached_values = [generation.native_tokens_cached for generation in generations]
    nonzero_values = [value for value in cached_values if value > 0]
    if not nonzero_values:
        return CanaryAssessment(
            passed=False,
            baseline_cached_tokens=0,
            max_cached_tokens=0,
            history_growth_tokens=0,
            reason="cache_never_activated",
        )
    baseline = nonzero_values[0]
    maximum = max(nonzero_values)
    growth = maximum - baseline
    if growth >= min_history_growth:
        return CanaryAssessment(
            passed=True,
            baseline_cached_tokens=baseline,
            max_cached_tokens=maximum,
            history_growth_tokens=growth,
            reason="conversation_history_cache_grew",
        )
    return CanaryAssessment(
        passed=False,
        baseline_cached_tokens=baseline,
        max_cached_tokens=maximum,
        history_growth_tokens=growth,
        reason="conversation_history_cache_did_not_grow",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-5.4-mini")
    parser.add_argument("--min-history-growth", type=int, default=256)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing required environment variable `OPENROUTER_API_KEY`.")
    with _working_directory(keep_temp=args.keep_temp) as working_directory:
        result = run_canary(
            working_directory=working_directory,
            api_key=api_key,
            model=args.model,
            min_history_growth=args.min_history_growth,
        )
        rendered = json.dumps(result.to_dict(), indent=2)
        if args.json_out:
            Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0 if result.assessment.passed else 1


def run_canary(
    *,
    working_directory: Path,
    api_key: str,
    model: str,
    min_history_growth: int = 256,
) -> CanaryResult:
    candidate_path = working_directory / "candidate.py"
    stdout_path = working_directory / "claude.stdout.jsonl"
    build_canary_candidate(candidate_path)
    config_dir = working_directory / ".claude-canary-config"
    config_dir.mkdir(parents=True, exist_ok=True)

    with _openrouter_proxy() as base_url:
        env = {
            **os.environ,
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_AUTH_TOKEN": api_key,
            "ANTHROPIC_API_KEY": "",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CONFIG_DIR": str(config_dir),
        }
        completed = subprocess.run(
            [
                "claude",
                "-p",
                "--dangerously-skip-permissions",
                "--verbose",
                "--output-format",
                "stream-json",
                "--setting-sources",
                "local",
                "--max-turns",
                "8",
                "--model",
                model,
                build_canary_instruction(),
            ],
            cwd=working_directory,
            capture_output=True,
            text=True,
            env=env,
        )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Cache canary Claude invocation failed with code {completed.returncode}:\n"
            f"{completed.stderr}"
        )

    generation_ids = collect_generation_ids_from_stdout(completed.stdout)
    generations, missing_generation_ids = fetch_canary_generations(
        generation_ids,
        api_key=api_key,
    )
    assessment = assess_cache_growth(generations, min_history_growth=min_history_growth)
    return CanaryResult(
        claude_code_version=_extract_claude_code_version(completed.stdout),
        model=model,
        generation_ids=generation_ids,
        missing_generation_ids=missing_generation_ids,
        cached_sequence=tuple(generation.native_tokens_cached for generation in generations),
        prompt_sequence=tuple(generation.native_tokens_prompt for generation in generations),
        assessment=assessment,
        stdout_path=str(stdout_path),
        working_directory=str(working_directory),
    )


def fetch_canary_generations(
    generation_ids: tuple[str, ...],
    *,
    api_key: str,
) -> tuple[list[GenerationUsage], tuple[str, ...]]:
    generations: list[GenerationUsage] = []
    missing_generation_ids: list[str] = []
    for generation_id in generation_ids:
        try:
            generations.append(
                fetch_generation_usage(
                    generation_id,
                    api_key=api_key,
                    max_attempts=6,
                    retry_delay_seconds=3.0,
                )
            )
        except error.HTTPError as exc:
            if exc.code != 404:
                raise
            missing_generation_ids.append(generation_id)
    return generations, tuple(missing_generation_ids)


def _extract_claude_code_version(stdout_text: str) -> str | None:
    for raw_line in stdout_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") != "system" or payload.get("subtype") != "init":
            continue
        version = payload.get("claude_code_version")
        return str(version) if isinstance(version, str) else None
    return None


@contextmanager
def _working_directory(*, keep_temp: bool) -> Path:
    if keep_temp:
        root = Path(tempfile.mkdtemp(prefix="openrouter-cache-canary-"))
        yield root
        return
    with tempfile.TemporaryDirectory(prefix="openrouter-cache-canary-") as temp_dir:
        yield Path(temp_dir)


@contextmanager
def _openrouter_proxy() -> str:
    port = _reserve_loopback_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scaling_evolve.providers.agent.drivers.openrouter_proxy",
            "--listen-port",
            str(port),
            "--target-base-url",
            "https://openrouter.ai/api",
            "--pinned-provider",
            "OpenAI",
        ],
        cwd=Path.cwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        _wait_for_proxy(port)
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_proxy(port: int) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise RuntimeError(f"OpenRouter cache canary proxy on port {port} did not become ready.")


if __name__ == "__main__":
    raise SystemExit(main())
