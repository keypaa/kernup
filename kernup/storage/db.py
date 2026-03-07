"""SQLite helpers for Kernup run persistence."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunRecord:
    id: str
    model_id: str
    timestamp: str
    generation: int
    block_size: int
    num_warps: int
    num_stages: int
    kv_strategy: str
    split_k: int
    mutation_type: str


@dataclass(frozen=True)
class ResultRecord:
    id: str
    run_id: str
    tok_s: float
    ttft_ms: float
    latency_ms: float
    vram_used_gb: float
    is_best: int
    notes: str


def open_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with row dict support."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Create required tables when they do not already exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id            TEXT PRIMARY KEY,
            model_id      TEXT NOT NULL DEFAULT '',
            timestamp     TEXT NOT NULL,
            generation    INTEGER NOT NULL,
            block_size    INTEGER NOT NULL,
            num_warps     INTEGER NOT NULL,
            num_stages    INTEGER NOT NULL,
            kv_strategy   TEXT NOT NULL,
            split_k       INTEGER NOT NULL,
            mutation_type TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS results (
            id            TEXT PRIMARY KEY,
            run_id        TEXT NOT NULL,
            tok_s         REAL NOT NULL,
            ttft_ms       REAL NOT NULL,
            latency_ms    REAL NOT NULL,
            vram_used_gb  REAL NOT NULL,
            is_best       INTEGER NOT NULL,
            notes         TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        );
        """
    )

    # Backward-compatible migration for existing databases created before model_id was introduced.
    columns = [row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()]
    if "model_id" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN model_id TEXT NOT NULL DEFAULT ''")

    conn.commit()


def insert_run(conn: sqlite3.Connection, record: RunRecord) -> None:
    conn.execute(
        """
        INSERT INTO runs (id, model_id, timestamp, generation, block_size, num_warps, num_stages,
                          kv_strategy, split_k, mutation_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.id,
            record.model_id,
            record.timestamp,
            record.generation,
            record.block_size,
            record.num_warps,
            record.num_stages,
            record.kv_strategy,
            record.split_k,
            record.mutation_type,
        ),
    )
    conn.commit()


def insert_result(conn: sqlite3.Connection, record: ResultRecord) -> None:
    conn.execute(
        """
        INSERT INTO results (id, run_id, tok_s, ttft_ms, latency_ms, vram_used_gb, is_best, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.id,
            record.run_id,
            record.tok_s,
            record.ttft_ms,
            record.latency_ms,
            record.vram_used_gb,
            record.is_best,
            record.notes,
        ),
    )
    conn.commit()


def list_results_for_run(conn: sqlite3.Connection, run_id: str) -> list[ResultRecord]:
    cursor = conn.execute(
        """
        SELECT id, run_id, tok_s, ttft_ms, latency_ms, vram_used_gb, is_best, notes
        FROM results
        WHERE run_id = ?
        ORDER BY id ASC
        """,
        (run_id,),
    )
    rows = cursor.fetchall()
    return [
        ResultRecord(
            id=row["id"],
            run_id=row["run_id"],
            tok_s=row["tok_s"],
            ttft_ms=row["ttft_ms"],
            latency_ms=row["latency_ms"],
            vram_used_gb=row["vram_used_gb"],
            is_best=row["is_best"],
            notes=row["notes"],
        )
        for row in rows
    ]
