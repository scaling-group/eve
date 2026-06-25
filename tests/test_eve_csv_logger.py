from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from scaling_evolve.algorithms.eve.logger.csv import CSVEveLogger
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.workflow.phase2 import Phase2Result


def _usage(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    agent_turns: int = 1,
    model_cost_usd: float = 0.0,
    wallclock_seconds: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            agent_turns=agent_turns,
            model_cost_usd=model_cost_usd,
            wallclock_seconds=wallclock_seconds,
        )
    )


def _phase2_result(
    *,
    solver_id: str,
    optimizer_id: str,
    score: float,
    elo: float,
    workspace_id: str,
    usage: SimpleNamespace,
) -> Phase2Result:
    return Phase2Result(
        optimizer=PopulationEntry(id="optimizer-parent", files={}, score={"elo": 1400.0}, logs={}),
        produced_solver=PopulationEntry(
            id=solver_id,
            files={},
            score={"score": score},
            logs={},
        ),
        produced_optimizer=PopulationEntry(
            id=optimizer_id,
            files={},
            score={"elo": elo},
            logs={},
        ),
        rollouts=[usage],
        workspace_id=workspace_id,
        worker_name="normal",
    )


def test_csv_eve_logger_writes_metrics_and_candidate_csvs(tmp_path) -> None:
    logger = CSVEveLogger(
        run_id="run-123",
        full_config={"run_root": str(tmp_path)},
        enabled=True,
        output_dir=str(tmp_path / "telemetry"),
        excluded_score_fields=["info"],
    )

    phase2_results = [
        Phase2Result(
            optimizer=PopulationEntry(id="optimizer-a", files={}, score={"elo": 1400.0}, logs={}),
            sampled_solvers=[],
            prefill_solver=None,
            produced_solver=PopulationEntry(
                id="solver-a",
                files={},
                score={
                    "score": 1.25,
                    "info": {"target": 3.0},
                    "comment": "keep me in json",
                },
                logs={},
            ),
            produced_optimizer=PopulationEntry(
                id="optimizer-phase2",
                files={"APPROACH.md": "new"},
                score={"elo": 1412.0, "note": "phase2 optimizer"},
                logs={},
            ),
            rollouts=[
                SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=10,
                        output_tokens=5,
                        cache_read_tokens=1,
                        cache_creation_tokens=0,
                        agent_turns=2,
                        model_cost_usd=0.4,
                        wallclock_seconds=4.0,
                    )
                )
            ],
            workspace_id="workspace-a",
        )
    ]
    logger.on_iteration(
        iteration=3,
        solver_entries=[PopulationEntry(id="solver-a", files={}, score={"score": 1.25}, logs={})],
        optimizer_entries=[
            PopulationEntry(id="optimizer-new", files={}, score={"elo": 1425.0}, logs={})
        ],
        phase2_results=phase2_results,
    )
    logger.finish(
        solver_entries=[PopulationEntry(id="solver-a", files={}, score={"score": 1.25}, logs={})],
        optimizer_entries=[
            PopulationEntry(id="optimizer-new", files={}, score={"elo": 1425.0}, logs={})
        ],
        iterations_completed=8,
    )

    metrics_path = tmp_path / "telemetry" / "iteration_metrics.csv"
    phase2_solver_path = tmp_path / "telemetry" / "phase2_solvers.csv"
    phase2_optimizer_path = tmp_path / "telemetry" / "phase2_optimizers.csv"
    summary_path = tmp_path / "telemetry" / "summary.json"

    assert metrics_path.exists()
    assert phase2_solver_path.exists()
    assert phase2_optimizer_path.exists()
    assert summary_path.exists()

    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metrics_rows = list(csv.DictReader(handle))
    assert len(metrics_rows) == 1
    assert metrics_rows[0]["population/solver_size"] == "1"
    assert metrics_rows[0]["usage/iteration/model_cost_usd"] == "0.4"
    assert metrics_rows[0]["phase2/iteration/max/score"] == "1.25"
    assert metrics_rows[0]["phase2/scores/score"] == "[1.25]"

    with phase2_solver_path.open(encoding="utf-8", newline="") as handle:
        phase2_solver_rows = list(csv.DictReader(handle))
    assert len(phase2_solver_rows) == 1
    assert phase2_solver_rows[0]["worker_index"] == "0"
    assert phase2_solver_rows[0]["solver_id"] == "solver-a"
    assert json.loads(phase2_solver_rows[0]["score_json"]) == {
        "comment": "keep me in json",
        "info": {"target": 3.0},
        "score": 1.25,
    }

    with phase2_optimizer_path.open(encoding="utf-8", newline="") as handle:
        phase2_optimizer_rows = list(csv.DictReader(handle))
    assert len(phase2_optimizer_rows) == 1
    assert phase2_optimizer_rows[0]["optimizer_id"] == "optimizer-phase2"
    assert json.loads(phase2_optimizer_rows[0]["score_json"]) == {
        "elo": 1412.0,
        "note": "phase2 optimizer",
    }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_iterations"] == 8
    assert summary["best_solver_score"] == {"score": 1.25}
    assert summary["best_solver_first_seen_iteration"] == 3
    assert summary["best_solver_id"] == "solver-a"
    assert summary["best_optimizer_score"] == {"elo": 1425.0}
    assert summary["best_optimizer_first_seen_iteration"] == 3
    assert summary["best_optimizer_id"] == "optimizer-new"


def test_csv_eve_logger_updates_summary_each_iteration(tmp_path) -> None:
    logger = CSVEveLogger(
        run_id="run-123",
        full_config={"run_root": str(tmp_path)},
        enabled=True,
        output_dir=str(tmp_path / "telemetry"),
        excluded_score_fields=["info"],
    )

    logger.on_iteration(
        iteration=3,
        solver_entries=[PopulationEntry(id="solver-a", files={}, score={"score": 1.25}, logs={})],
        optimizer_entries=[
            PopulationEntry(id="optimizer-new", files={}, score={"elo": 1425.0}, logs={})
        ],
        phase2_results=[],
    )

    summary = json.loads((tmp_path / "telemetry" / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_iterations"] == 3
    assert summary["best_solver_score"] == {"score": 1.25}
    assert summary["best_optimizer_score"] == {"elo": 1425.0}
    assert summary["best_solver_first_seen_iteration"] == 3
    assert summary["best_solver_id"] == "solver-a"
    assert summary["best_optimizer_first_seen_iteration"] == 3
    assert summary["best_optimizer_id"] == "optimizer-new"


def test_csv_eve_logger_summary_keeps_earliest_best_entry_on_ties(tmp_path) -> None:
    logger = CSVEveLogger(
        run_id="run-123",
        full_config={"run_root": str(tmp_path)},
        enabled=True,
        output_dir=str(tmp_path / "telemetry"),
        excluded_score_fields=[],
    )

    logger.on_iteration(
        iteration=2,
        solver_entries=[
            PopulationEntry(id="solver-early", files={}, score={"score": 3.0}, logs={}),
            PopulationEntry(id="solver-low", files={}, score={"score": 1.0}, logs={}),
        ],
        optimizer_entries=[
            PopulationEntry(id="optimizer-early", files={}, score={"elo": 1500.0}, logs={}),
        ],
        phase2_results=[],
    )
    logger.on_iteration(
        iteration=5,
        solver_entries=[
            PopulationEntry(id="solver-late", files={}, score={"score": 3.0}, logs={}),
            PopulationEntry(id="solver-low", files={}, score={"score": 2.0}, logs={}),
        ],
        optimizer_entries=[
            PopulationEntry(id="optimizer-late", files={}, score={"elo": 1500.0}, logs={}),
        ],
        phase2_results=[],
    )

    summary = json.loads((tmp_path / "telemetry" / "summary.json").read_text(encoding="utf-8"))
    assert summary["best_solver_score"] == {"score": 3.0}
    assert summary["best_solver_first_seen_iteration"] == 2
    assert summary["best_solver_id"] == "solver-early"
    assert summary["best_optimizer_score"] == {"elo": 1500.0}
    assert summary["best_optimizer_first_seen_iteration"] == 2
    assert summary["best_optimizer_id"] == "optimizer-early"


def test_csv_eve_logger_restore_for_resume_truncates_and_hydrates_state(tmp_path) -> None:
    telemetry = tmp_path / "telemetry"
    logger = CSVEveLogger(
        run_id="run-123",
        full_config={"run_root": str(tmp_path)},
        enabled=True,
        output_dir=str(telemetry),
        excluded_score_fields=[],
    )
    logger.on_iteration(
        iteration=1,
        solver_entries=[PopulationEntry(id="solver-1", files={}, score={"score": 1.0}, logs={})],
        optimizer_entries=[
            PopulationEntry(id="optimizer-1", files={}, score={"elo": 1400.0}, logs={})
        ],
        phase2_results=[
            _phase2_result(
                solver_id="solver-1",
                optimizer_id="optimizer-1",
                score=1.0,
                elo=1400.0,
                workspace_id="step_1_a",
                usage=_usage(input_tokens=10, output_tokens=2, model_cost_usd=0.1),
            )
        ],
    )
    logger.on_iteration(
        iteration=2,
        solver_entries=[PopulationEntry(id="solver-2", files={}, score={"score": 2.0}, logs={})],
        optimizer_entries=[
            PopulationEntry(id="optimizer-2", files={}, score={"elo": 1500.0}, logs={})
        ],
        phase2_results=[
            _phase2_result(
                solver_id="solver-2",
                optimizer_id="optimizer-2",
                score=2.0,
                elo=1500.0,
                workspace_id="step_2_failed",
                usage=_usage(input_tokens=20, output_tokens=4, model_cost_usd=0.2),
            )
        ],
    )

    resumed = CSVEveLogger(
        run_id="run-123",
        full_config={"run_root": str(tmp_path)},
        enabled=True,
        output_dir=str(telemetry),
        excluded_score_fields=[],
        resume_anchor_iteration=1,
    )

    with (telemetry / "iteration_metrics.csv").open(encoding="utf-8", newline="") as handle:
        metrics_rows = list(csv.DictReader(handle))
    assert [row["iteration"] for row in metrics_rows] == ["1"]
    assert metrics_rows[0]["usage/cumulative/model_cost_usd"] == "0.1"
    assert metrics_rows[0]["phase2/cumulative/max/score"] == "1.0"

    with (telemetry / "phase2_solvers.csv").open(encoding="utf-8", newline="") as handle:
        solver_rows = list(csv.DictReader(handle))
    assert [row["workspace_id"] for row in solver_rows] == ["step_1_a"]

    resumed.finish(
        solver_entries=[
            PopulationEntry(id="solver-1", files={}, score={"score": 1.0}, logs={}),
            PopulationEntry(id="solver-2", files={}, score={"score": 2.0}, logs={}),
        ],
        optimizer_entries=[
            PopulationEntry(id="optimizer-1", files={}, score={"elo": 1400.0}, logs={}),
            PopulationEntry(id="optimizer-2", files={}, score={"elo": 1500.0}, logs={}),
        ],
        iterations_completed=1,
    )
    summary = json.loads((telemetry / "summary.json").read_text(encoding="utf-8"))
    assert summary["best_solver_id"] == "solver-1"
    assert summary["best_solver_first_seen_iteration"] == 1
    assert summary["best_optimizer_id"] == "optimizer-1"
    assert summary["best_optimizer_first_seen_iteration"] == 1

    resumed.on_iteration(
        iteration=2,
        solver_entries=[PopulationEntry(id="solver-3", files={}, score={"score": 3.0}, logs={})],
        optimizer_entries=[
            PopulationEntry(id="optimizer-3", files={}, score={"elo": 1600.0}, logs={})
        ],
        phase2_results=[
            _phase2_result(
                solver_id="solver-3",
                optimizer_id="optimizer-3",
                score=3.0,
                elo=1600.0,
                workspace_id="step_2_resumed",
                usage=_usage(input_tokens=30, output_tokens=6, model_cost_usd=0.3),
            )
        ],
    )
    with (telemetry / "iteration_metrics.csv").open(encoding="utf-8", newline="") as handle:
        resumed_metrics_rows = list(csv.DictReader(handle))
    assert [row["iteration"] for row in resumed_metrics_rows] == ["1", "2"]
    assert resumed_metrics_rows[1]["usage/cumulative/model_cost_usd"] == "0.4"
    assert resumed_metrics_rows[1]["phase2/cumulative/max/score"] == "3.0"

    with (telemetry / "phase2_solvers.csv").open(encoding="utf-8", newline="") as handle:
        resumed_solver_rows = list(csv.DictReader(handle))
    assert [row["workspace_id"] for row in resumed_solver_rows] == [
        "step_1_a",
        "step_2_resumed",
    ]


def test_csv_eve_logger_restore_reuses_safe_best_record_from_post_anchor_summary(
    tmp_path,
) -> None:
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    (telemetry / "summary.json").write_text(
        json.dumps(
            {
                "total_iterations": 2,
                "best_solver_id": "solver-anchor-best",
                "best_solver_score": {"score": 5.0},
                "best_solver_first_seen_iteration": 1,
                "best_optimizer_id": "optimizer-anchor-best",
                "best_optimizer_score": {"elo": 1550.0},
                "best_optimizer_first_seen_iteration": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    resumed = CSVEveLogger(
        run_id="run-123",
        full_config={"run_root": str(tmp_path)},
        enabled=True,
        output_dir=str(telemetry),
        excluded_score_fields=[],
        resume_anchor_iteration=1,
    )
    resumed.finish(
        solver_entries=[
            PopulationEntry(id="solver-fallback", files={}, score={"score": 1.0}, logs={})
        ],
        optimizer_entries=[
            PopulationEntry(id="optimizer-fallback", files={}, score={"elo": 1400.0}, logs={})
        ],
        iterations_completed=1,
    )

    summary = json.loads((telemetry / "summary.json").read_text(encoding="utf-8"))
    assert summary["best_solver_id"] == "solver-anchor-best"
    assert summary["best_solver_first_seen_iteration"] == 1
    assert summary["best_optimizer_id"] == "optimizer-anchor-best"
    assert summary["best_optimizer_first_seen_iteration"] == 1


def test_csv_eve_logger_writes_resume_anchor_summary_immediately(tmp_path) -> None:
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    (telemetry / "summary.json").write_text(
        json.dumps(
            {
                "total_iterations": 2,
                "best_solver_id": "solver-anchor-best",
                "best_solver_score": {"score": 5.0},
                "best_solver_first_seen_iteration": 1,
                "best_optimizer_id": "optimizer-anchor-best",
                "best_optimizer_score": {"elo": 1550.0},
                "best_optimizer_first_seen_iteration": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    logger = CSVEveLogger(
        run_id="run-123",
        full_config={"run_root": str(tmp_path)},
        enabled=True,
        output_dir=str(telemetry),
        excluded_score_fields=[],
        resume_anchor_iteration=1,
    )
    logger.write_resume_anchor_summary(
        solver_entries=[
            PopulationEntry(id="solver-fallback", files={}, score={"score": 1.0}, logs={})
        ],
        optimizer_entries=[
            PopulationEntry(id="optimizer-fallback", files={}, score={"elo": 1400.0}, logs={})
        ],
        iterations_completed=1,
    )

    summary = json.loads((telemetry / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_iterations"] == 1
    assert summary["best_solver_id"] == "solver-anchor-best"
    assert summary["best_solver_first_seen_iteration"] == 1
    assert summary["best_optimizer_id"] == "optimizer-anchor-best"
    assert summary["best_optimizer_first_seen_iteration"] == 1
