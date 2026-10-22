"""Results store: one sqlite DB where every run is saved with its reproducibility envelope.

A run is rejected unless it carries git_sha, dataset_sha256, seed, and model_key, so
every stored number can be regenerated from a known commit, dataset, and seed.
Remaining envelope fields are kept as JSON config.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from syco import paths

REQUIRED_FIELDS = ("git_sha", "dataset_sha256", "seed", "model_key")
_META_KEYS = {"_meta", *REQUIRED_FIELDS}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    git_sha TEXT NOT NULL,
    dataset_sha256 TEXT NOT NULL,
    seed INTEGER NOT NULL,
    model_key TEXT NOT NULL,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    name TEXT NOT NULL,
    value REAL NOT NULL,
    ci_low REAL,
    ci_high REAL,
    PRIMARY KEY (run_id, name)
);
"""


def connect(path: str | Path = paths.RESULTS_DB) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _normalize_metric(metric: float | dict) -> dict:
    if isinstance(metric, dict):
        return {"value": metric["value"], "ci_low": metric.get("ci_low"), "ci_high": metric.get("ci_high")}
    return {"value": metric, "ci_low": None, "ci_high": None}


def record_run(conn: sqlite3.Connection, envelope: dict, metrics: dict) -> int:
    """Store a run and its metrics atomically; return the run id."""
    missing = [f for f in REQUIRED_FIELDS if f not in envelope]
    if missing:
        raise ValueError(f"envelope missing required field(s): {', '.join(missing)}")
    config = {k: v for k, v in envelope.items() if k not in _META_KEYS}
    with conn:
        cur = conn.execute(
            "INSERT INTO runs (created_at, git_sha, dataset_sha256, seed, model_key, config_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                envelope["git_sha"],
                envelope["dataset_sha256"],
                envelope["seed"],
                envelope["model_key"],
                json.dumps(config, sort_keys=True),
            ),
        )
        run_id = cur.lastrowid
        for name, metric in metrics.items():
            m = _normalize_metric(metric)
            conn.execute(
                "INSERT INTO metrics (run_id, name, value, ci_low, ci_high) VALUES (?, ?, ?, ?, ?)",
                (run_id, name, m["value"], m["ci_low"], m["ci_high"]),
            )
    return run_id


def fetch_runs(conn: sqlite3.Connection, model_key: str | None = None) -> list[dict]:
    query = "SELECT * FROM runs" + (" WHERE model_key = ?" if model_key else "") + " ORDER BY id"
    rows = conn.execute(query, (model_key,) if model_key else ()).fetchall()
    runs = []
    for row in rows:
        metric_rows = conn.execute(
            "SELECT name, value, ci_low, ci_high FROM metrics WHERE run_id = ?", (row["id"],)
        ).fetchall()
        runs.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "git_sha": row["git_sha"],
                "dataset_sha256": row["dataset_sha256"],
                "seed": row["seed"],
                "model_key": row["model_key"],
                "config": json.loads(row["config_json"]),
                "metrics": {
                    m["name"]: {"value": m["value"], "ci_low": m["ci_low"], "ci_high": m["ci_high"]}
                    for m in metric_rows
                },
            }
        )
    return runs


EXPORT_CONFIG_FIELDS = (
    "adapter", "samples", "judge_model", "judge_rubric_version", "stance_model", "stance_prompt_version",
)


def export_runs_csv(conn: sqlite3.Connection, path: str | Path) -> None:
    """Write every run's metrics as one long CSV (one row per run x metric) for sharing.

    results.db stays local; this file is the committed, reviewable copy of the numbers.
    """
    fields = ["run_id", "created_at", "model_key", "seed", "git_sha", "dataset_sha256",
              *EXPORT_CONFIG_FIELDS, "metric", "value", "ci_low", "ci_high"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in fetch_runs(conn):
            base = {
                "run_id": run["id"], "created_at": run["created_at"], "model_key": run["model_key"],
                "seed": run["seed"], "git_sha": run["git_sha"], "dataset_sha256": run["dataset_sha256"],
                **{k: run["config"].get(k, "") for k in EXPORT_CONFIG_FIELDS},
            }
            for name, metric in sorted(run["metrics"].items()):
                writer.writerow({**base, "metric": name, **metric})
