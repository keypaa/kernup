"""Export helpers for run artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def export_results_json(conn: sqlite3.Connection, run_id: str, output_path: str | Path) -> Path:
    """Export all rows for a run id to a JSON file."""
    run_row = conn.execute(
        """
        SELECT id, model_id, timestamp, generation, block_size, num_warps, num_stages, kv_strategy,
               split_k, mutation_type
        FROM runs
        WHERE id = ?
        """,
        (run_id,),
    ).fetchone()

    if run_row is None:
        raise ValueError(f"Run id not found: {run_id}")

    result_rows = conn.execute(
        """
        SELECT id, run_id, tok_s, ttft_ms, latency_ms, vram_used_gb, is_best, notes
        FROM results
        WHERE run_id = ?
        ORDER BY id ASC
        """,
        (run_id,),
    ).fetchall()

    payload = {
        "run": dict(run_row),
        "results": [dict(row) for row in result_rows],
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
