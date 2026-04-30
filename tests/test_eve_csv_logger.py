from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from scaling_evolve.algorithms.eve.logger.csv import CSVEveLogger
from scaling_evolve.algorithms.eve.populations.entry import PopulationEntry
from scaling_evolve.algorithms.eve.workflow.phase2 import Phase2Result
from scaling_evolve.algorithms.eve.workflow.phase4 import Phase4Result


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
    phase4_results = [
        Phase4Result(
            lead_optimizer=PopulationEntry(
                id="optimizer-lead",
                files={"APPROACH.md": "lead"},
                score={"elo": 1500.0},
                logs={},
            ),
            sampled_optimizers=[],
            prefill_optimizer=PopulationEntry(
                id="optimizer-prefill",
                files={"APPROACH.md": "prefill"},
                score={"elo": 1425.0},
                logs={},
            ),
            produced_optimizer=PopulationEntry(
                id="optimizer-new",
                files={"APPROACH.md": "new"},
                score={"elo": 1425.0, "note": "keep raw json"},
                logs={},
            ),
            rollouts=[
                SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=7,
                        output_tokens=3,
                        cache_read_tokens=0,
                        cache_creation_tokens=1,
                        agent_turns=1,
                        model_cost_usd=0.75,
                        wallclock_seconds=2.0,
                    )
                )
            ],
            workspace_id="workspace-123",
        )
    ]

    logger.on_iteration(
        iteration=3,
        solver_entries=[PopulationEntry(id="solver-a", files={}, score={"score": 1.25}, logs={})],
        optimizer_entries=[
            PopulationEntry(id="optimizer-new", files={}, score={"elo": 1425.0}, logs={})
        ],
        phase2_results=phase2_results,
        phase4_results=phase4_results,
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
    phase4_optimizer_path = tmp_path / "telemetry" / "phase4_optimizers.csv"
    summary_path = tmp_path / "telemetry" / "summary.json"

    assert metrics_path.exists()
    assert phase2_solver_path.exists()
    assert phase2_optimizer_path.exists()
    assert phase4_optimizer_path.exists()
    assert summary_path.exists()

    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metrics_rows = list(csv.DictReader(handle))
    assert len(metrics_rows) == 1
    assert metrics_rows[0]["population/solver_size"] == "1"
    assert metrics_rows[0]["usage/iteration/model_cost_usd"] == "1.15"
    assert metrics_rows[0]["phase2/iteration/max/score"] == "1.25"
    assert metrics_rows[0]["phase4/iteration/max/elo"] == "1425.0"
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

    with phase4_optimizer_path.open(encoding="utf-8", newline="") as handle:
        phase4_optimizer_rows = list(csv.DictReader(handle))
    assert len(phase4_optimizer_rows) == 1
    assert phase4_optimizer_rows[0]["optimizer_id"] == "optimizer-new"
    assert json.loads(phase4_optimizer_rows[0]["score_json"]) == {
        "elo": 1425.0,
        "note": "keep raw json",
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
        phase4_results=[],
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
        phase4_results=[],
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
        phase4_results=[],
    )

    summary = json.loads((tmp_path / "telemetry" / "summary.json").read_text(encoding="utf-8"))
    assert summary["best_solver_score"] == {"score": 3.0}
    assert summary["best_solver_first_seen_iteration"] == 2
    assert summary["best_solver_id"] == "solver-early"
    assert summary["best_optimizer_score"] == {"elo": 1500.0}
    assert summary["best_optimizer_first_seen_iteration"] == 2
    assert summary["best_optimizer_id"] == "optimizer-early"
