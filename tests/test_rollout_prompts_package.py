from __future__ import annotations

from pathlib import Path

from scaling_evolve.algorithms.eve.rollout_prompts.default import (
    BudgetPrompt,
    PromptContext,
    StaticRolloutPrompt,
)
from scaling_evolve.providers.agent.turns import inspect_transcript_turn_state


def test_inspect_transcript_turn_state_for_claude_payloads(tmp_path: Path) -> None:
    transcript_path = tmp_path / "claude.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                (
                    '{"type":"assistant","requestId":"req-1","message":{"id":"msg-1","content":'
                    '[{"type":"text","text":"thinking"},{"type":"tool_use","id":"call-a",'
                    '"name":"Read","input":{"file_path":"candidate.py"}},{"type":"tool_use",'
                    '"id":"call-b","name":"Bash","input":{"command":"ls"}}]}}'
                ),
                (
                    '{"type":"assistant","requestId":"req-1","message":{"id":"msg-1","content":'
                    '[{"type":"tool_use","id":"call-a","name":"Read","input":{}}]}}'
                ),
                (
                    '{"type":"assistant","requestId":"req-2","message":{"id":"msg-2","content":'
                    '[{"type":"text","text":"final"}]}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = inspect_transcript_turn_state(transcript_path)

    assert state.format_name == "claude"
    assert state.turn_count == 1
    assert state.latest_batch_tool_ids == ("call-a",)


def test_inspect_transcript_turn_state_for_codex_exec_payloads(tmp_path: Path) -> None:
    transcript_path = tmp_path / "codex-exec.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                '{"type":"thread.started","thread_id":"session-1"}',
                '{"type":"item.completed","item":{"type":"agent_message","text":"Inspecting"}}',
                (
                    '{"type":"item.completed","item":{"type":"function_call","name":"bash",'
                    '"call_id":"call-1"}}'
                ),
                (
                    '{"type":"item.completed","item":{"type":"function_call","name":"bash",'
                    '"call_id":"call-2"}}'
                ),
                '{"type":"turn.completed","usage":{"input_tokens":10}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = inspect_transcript_turn_state(transcript_path)

    assert state.format_name == "codex_exec"
    assert state.turn_count == 1
    assert state.latest_batch_tool_ids == ("call-1", "call-2")


def test_inspect_transcript_turn_state_for_codex_tmux_payloads(tmp_path: Path) -> None:
    transcript_path = tmp_path / "codex-tmux.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                '{"type":"session_meta","payload":{"id":"session-1"}}',
                (
                    '{"type":"event_msg","payload":{"type":"agent_message",'
                    '"message":"Inspecting the workspace."}}'
                ),
                (
                    '{"type":"response_item","payload":{"type":"function_call","name":"exec_command",'
                    '"arguments":"{}","call_id":"call-1"}}'
                ),
                (
                    '{"type":"response_item","payload":{"type":"function_call","name":"exec_command",'
                    '"arguments":"{}","call_id":"call-2"}}'
                ),
                (
                    '{"type":"response_item","payload":{"type":"function_call_output","call_id":"call-1",'
                    '"output":"ok"}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = inspect_transcript_turn_state(transcript_path)

    assert state.format_name == "codex_tmux"
    assert state.turn_count == 1
    assert state.latest_batch_tool_ids == ("call-1", "call-2")


def test_budget_prompt_user_renders_announcement_from_context() -> None:
    prompt = BudgetPrompt()
    text = prompt.user(PromptContext(workspace=Path("."), rollout_max_turns=10))

    assert text == (
        "Turn budget enabled: this session has 10 turns per rollout. "
        "After each turn you will see `[Budget] N/10 turns remaining`. "
        "Use that signal to pace your work - the current rollout will be terminated "
        "when the budget runs out."
    )


def test_budget_prompt_turn_renders_remaining_turns_from_context() -> None:
    prompt = BudgetPrompt()

    assert (
        prompt.turn(PromptContext(workspace=Path("."), rollout_max_turns=10, turns_remaining=7))
        == "[Budget] 7/10 turns remaining"
    )
    assert (
        prompt.turn(PromptContext(workspace=Path("."), rollout_max_turns=10, turns_remaining=1))
        == "[Budget] 1/10 turns remaining"
    )


def test_budget_prompt_exposes_turn_template_and_format_kwargs() -> None:
    prompt = BudgetPrompt()
    ctx = PromptContext(workspace=Path("."), rollout_max_turns=12)

    assert (
        prompt.turn_template_source()
        == "[Budget] {turns_remaining}/{rollout_max_turns} turns remaining"
    )
    assert prompt.turn_format_kwargs(ctx) == {"rollout_max_turns": 12}


def test_budget_prompt_system_is_reserved() -> None:
    prompt = BudgetPrompt()

    assert prompt.system(PromptContext(workspace=Path("."), rollout_max_turns=10)) is None


def test_static_rollout_prompt_supports_all_lifecycle_points(tmp_path: Path) -> None:
    static = StaticRolloutPrompt(
        system_text="system prompt",
        user_text="user prompt",
        turn_text="turn prompt",
    )
    ctx = PromptContext(workspace=tmp_path)

    assert static.system(ctx) == "system prompt"
    assert static.user(ctx) == "user prompt"
    assert static.turn(ctx) == "turn prompt"
    assert static.turn_template_source() == "turn prompt"
    assert static.turn_format_kwargs(ctx) == {}
