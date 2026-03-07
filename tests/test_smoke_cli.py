from click.testing import CliRunner
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from kernup.benchmark.runtime import RuntimeBenchmarkResult
from kernup.cli.main import cli
from kernup.storage.db import ResultRecord, RunRecord, create_schema, insert_result, insert_run, list_results_for_run, open_connection


def test_cli_help_shows_profile_command() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "profile" in result.output
    assert "optimize" in result.output
    assert "patch" in result.output
    assert "bench" in result.output


def test_profile_dry_run_with_gpu_bypass_succeeds() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            [
                "profile",
                "--hf",
                "Qwen/Qwen2.5-7B",
                "--dry-run",
                "--allow-no-gpu",
                "--output",
                "./kernup_results_test",
            ],
        )

        assert result.exit_code == 0
        assert "Bypassing GPU check" in result.output
        assert "Run ID:" in result.output
        assert "Dry-run complete" in result.output


def test_profile_without_dry_run_returns_placeholder_error() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["profile", "--hf", "Qwen/Qwen2.5-7B", "--allow-no-gpu"],
    )

    assert result.exit_code != 0
    assert "Full profiling is not implemented yet" in result.output


def test_profile_dry_run_writes_artifacts_and_export() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            [
                "profile",
                "--hf",
                "Qwen/Qwen2.5-7B",
                "--dry-run",
                "--allow-no-gpu",
                "--output",
                "./kernup_results",
                "--export",
            ],
        )

        assert result.exit_code == 0

        lines = [line.strip() for line in result.output.splitlines()]
        run_id_line = next(line for line in lines if line.startswith("Run ID:"))
        run_id = run_id_line.split(":", maxsplit=1)[1].strip()

        run_dir = Path("kernup_results") / run_id
        assert (run_dir / "kernup.db").exists()
        assert (run_dir / "kernup.log").exists()
        assert (run_dir / "profile.json").exists()
        assert (run_dir / "results_export.json").exists()


def test_storage_roundtrip_in_memory() -> None:
    with open_connection(":memory:") as conn:
        create_schema(conn)
        run_id = "run_test_001"
        insert_run(
            conn,
            RunRecord(
                id=run_id,
                model_id="Qwen/Qwen2.5-7B",
                timestamp="2026-03-06T12:00:00+00:00",
                generation=1,
                block_size=128,
                num_warps=4,
                num_stages=2,
                kv_strategy="full",
                split_k=1,
                mutation_type="light",
            ),
        )
        insert_result(
            conn,
            ResultRecord(
                id=str(uuid4()),
                run_id=run_id,
                tok_s=100.0,
                ttft_ms=120.0,
                latency_ms=35.0,
                vram_used_gb=10.5,
                is_best=1,
                notes="test",
            ),
        )

        rows = list_results_for_run(conn, run_id)
        assert len(rows) == 1
        assert rows[0].tok_s == 100.0


def test_optimize_phase1_dry_run_succeeds() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            [
                "optimize",
                "--hf",
                "Qwen/Qwen2.5-7B",
                "--phase",
                "1",
                "--dry-run",
                "--allow-no-gpu",
                "--iterations",
                "6",
                "--population",
                "4",
                "--output",
                "./kernup_results",
            ],
        )

        assert result.exit_code == 0
        assert "Phase 1 dry-run complete" in result.output
        assert "Best tok/s:" in result.output


def test_optimize_phase2_dry_run_succeeds() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            [
                "optimize",
                "--hf",
                "Qwen/Qwen2.5-7B",
                "--phase",
                "2",
                "--dry-run",
                "--allow-no-gpu",
                "--output",
                "./kernup_results",
            ],
        )

        assert result.exit_code == 0
        assert "Phase 2 dry-run complete" in result.output
        assert "Evolution artifact:" in result.output
        assert "Best tok/s:" in result.output


def test_optimize_real_run_rejects_gpu_bypass() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "optimize",
            "--hf",
            "Qwen/Qwen2.5-7B",
            "--phase",
            "1",
            "--allow-no-gpu",
        ],
    )

    assert result.exit_code != 0
    assert "Real optimization requires GPU" in result.output


def test_status_and_clean_commands() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        status_run_dir = Path("kernup_results") / "run_20260306_000001_abc123"
        status_run_dir.mkdir(parents=True)
        db_path = status_run_dir / "kernup.db"

        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE runs (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL DEFAULT '',
                    timestamp TEXT,
                    generation INTEGER,
                    block_size INTEGER,
                    num_warps INTEGER,
                    num_stages INTEGER,
                    kv_strategy TEXT,
                    split_k INTEGER,
                    mutation_type TEXT
                );
                CREATE TABLE results (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    tok_s REAL,
                    ttft_ms REAL,
                    latency_ms REAL,
                    vram_used_gb REAL,
                    is_best INTEGER,
                    notes TEXT
                );
                """
            )
            conn.execute(
                """
                INSERT INTO runs (id, model_id, timestamp, generation, block_size, num_warps, num_stages, kv_strategy, split_k, mutation_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run_20260306_000001_abc123",
                    "Qwen/Qwen2.5-7B",
                    "2026-03-06T00:00:00+00:00",
                    0,
                    0,
                    0,
                    0,
                    "n/a",
                    0,
                    "dry-run",
                ),
            )
            conn.execute(
                """
                INSERT INTO results (id, run_id, tok_s, ttft_ms, latency_ms, vram_used_gb, is_best, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (str(uuid4()), "run_20260306_000001_abc123", 12.3, 98.0, 45.0, 6.0, 1, "test"),
            )
            conn.commit()

        status = runner.invoke(cli, ["status", "--results", "./kernup_results"])
        assert status.exit_code == 0
        assert "Found 1 run(s)" in status.output
        assert "model=Qwen/Qwen2.5-7B" in status.output
        assert "best tok/s=12.300" in status.output

        status_filtered = runner.invoke(
            cli,
            ["status", "--results", "./kernup_results", "--hf", "Qwen/Qwen2.5-7B"],
        )
        assert status_filtered.exit_code == 0
        assert "run_20260306_000001_abc123" in status_filtered.output

        clean_run_dir = Path("kernup_to_clean") / "run_20260306_000002_def456"
        clean_run_dir.mkdir(parents=True)

        clean = runner.invoke(cli, ["clean", "--results", "./kernup_to_clean", "--yes"])
        assert clean.exit_code == 0
        assert "Removed 1 run(s)" in clean.output
        assert not clean_run_dir.exists()


def test_patch_and_bench_commands() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        run_dir = Path("kernup_results") / "run_20260306_000003_xyz987"
        run_dir.mkdir(parents=True)
        db_path = run_dir / "kernup.db"

        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE runs (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL DEFAULT '',
                    timestamp TEXT,
                    generation INTEGER,
                    block_size INTEGER,
                    num_warps INTEGER,
                    num_stages INTEGER,
                    kv_strategy TEXT,
                    split_k INTEGER,
                    mutation_type TEXT
                );
                CREATE TABLE results (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    tok_s REAL,
                    ttft_ms REAL,
                    latency_ms REAL,
                    vram_used_gb REAL,
                    is_best INTEGER,
                    notes TEXT
                );
                """
            )
            conn.execute(
                """
                INSERT INTO runs (id, model_id, timestamp, generation, block_size, num_warps, num_stages, kv_strategy, split_k, mutation_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run_20260306_000003_xyz987",
                    "Qwen/Qwen2.5-7B",
                    "2026-03-06T00:00:00+00:00",
                    0,
                    0,
                    0,
                    0,
                    "n/a",
                    0,
                    "dry-run",
                ),
            )
            conn.execute(
                """
                INSERT INTO results (id, run_id, tok_s, ttft_ms, latency_ms, vram_used_gb, is_best, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (str(uuid4()), "run_20260306_000003_xyz987", 33.3, 99.0, 44.0, 7.1, 1, "best"),
            )
            conn.commit()

        patch_result = runner.invoke(
            cli,
            [
                "patch",
                "--hf",
                "Qwen/Qwen2.5-7B",
                "--results",
                "./kernup_results",
                "--format",
                "simple",
                "--output",
                "./patch_out",
            ],
        )
        assert patch_result.exit_code == 0
        assert "Patch file:" in patch_result.output
        assert (Path("patch_out") / "patch_simple.py").exists()

        bench_result = runner.invoke(
            cli,
            [
                "bench",
                "--hf",
                "Qwen/Qwen2.5-7B",
                "--results",
                "./kernup_results",
                "--export",
                "--output",
                "./bench_out",
            ],
        )
        assert bench_result.exit_code == 0
        assert "Best tok/s: 33.300" in bench_result.output
        assert "Export:" in bench_result.output
        bench_exports = list(Path("bench_out").glob("bench_run_20260306_000003_xyz987.json"))
        assert len(bench_exports) == 1


def test_patch_smoke_succeeds_for_simple_patch() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        run_dir = Path("kernup_results") / "run_20260306_000009_smoke11"
        run_dir.mkdir(parents=True)
        db_path = run_dir / "kernup.db"

        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE runs (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL DEFAULT '',
                    timestamp TEXT,
                    generation INTEGER,
                    block_size INTEGER,
                    num_warps INTEGER,
                    num_stages INTEGER,
                    kv_strategy TEXT,
                    split_k INTEGER,
                    mutation_type TEXT
                );
                CREATE TABLE results (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    tok_s REAL,
                    ttft_ms REAL,
                    latency_ms REAL,
                    vram_used_gb REAL,
                    is_best INTEGER,
                    notes TEXT
                );
                """
            )
            conn.execute(
                """
                INSERT INTO runs (id, model_id, timestamp, generation, block_size, num_warps, num_stages, kv_strategy, split_k, mutation_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run_20260306_000009_smoke11",
                    "Qwen/Qwen2.5-7B",
                    "2026-03-06T00:00:00+00:00",
                    0,
                    0,
                    0,
                    0,
                    "n/a",
                    0,
                    "dry-run",
                ),
            )
            conn.execute(
                """
                INSERT INTO results (id, run_id, tok_s, ttft_ms, latency_ms, vram_used_gb, is_best, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (str(uuid4()), "run_20260306_000009_smoke11", 11.1, 99.0, 45.0, 6.0, 1, "best"),
            )
            conn.commit()

        patch_result = runner.invoke(
            cli,
            [
                "patch",
                "--hf",
                "Qwen/Qwen2.5-7B",
                "--results",
                "./kernup_results",
                "--format",
                "simple",
                "--output",
                "./patch_out",
                "--smoke",
            ],
        )
        assert patch_result.exit_code == 0
        assert "Smoke:" in patch_result.output


def test_patch_and_bench_ignore_incomplete_latest_run() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        valid_run = Path("kernup_results") / "run_20260306_000010_valid11"
        valid_run.mkdir(parents=True)
        db_path = valid_run / "kernup.db"

        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE runs (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL DEFAULT '',
                    timestamp TEXT,
                    generation INTEGER,
                    block_size INTEGER,
                    num_warps INTEGER,
                    num_stages INTEGER,
                    kv_strategy TEXT,
                    split_k INTEGER,
                    mutation_type TEXT
                );
                CREATE TABLE results (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    tok_s REAL,
                    ttft_ms REAL,
                    latency_ms REAL,
                    vram_used_gb REAL,
                    is_best INTEGER,
                    notes TEXT
                );
                """
            )
            conn.execute(
                """
                INSERT INTO runs (id, model_id, timestamp, generation, block_size, num_warps, num_stages, kv_strategy, split_k, mutation_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run_20260306_000010_valid11",
                    "Qwen/Qwen2.5-7B",
                    "2026-03-06T00:00:00+00:00",
                    0,
                    0,
                    0,
                    0,
                    "n/a",
                    0,
                    "dry-run",
                ),
            )
            conn.execute(
                """
                INSERT INTO results (id, run_id, tok_s, ttft_ms, latency_ms, vram_used_gb, is_best, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (str(uuid4()), "run_20260306_000010_valid11", 17.5, 88.0, 33.0, 5.4, 1, "best"),
            )
            conn.commit()

        # Create a newer incomplete run directory (no DB) to simulate interrupted job.
        incomplete = Path("kernup_results") / "run_20260306_000011_incomp1"
        incomplete.mkdir(parents=True)

        patch_result = runner.invoke(
            cli,
            [
                "patch",
                "--hf",
                "Qwen/Qwen2.5-7B",
                "--results",
                "./kernup_results",
                "--format",
                "simple",
                "--output",
                "./patch_out",
            ],
        )
        assert patch_result.exit_code == 0
        assert "run_20260306_000010_valid11" in patch_result.output

        bench_result = runner.invoke(
            cli,
            [
                "bench",
                "--hf",
                "Qwen/Qwen2.5-7B",
                "--results",
                "./kernup_results",
            ],
        )
        assert bench_result.exit_code == 0
        assert "Run: run_20260306_000010_valid11" in bench_result.output


def test_optimize_resume_reuses_latest_run_and_offsets_generations() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        first = runner.invoke(
            cli,
            [
                "optimize",
                "--hf",
                "Qwen/Qwen2.5-7B",
                "--phase",
                "1",
                "--dry-run",
                "--allow-no-gpu",
                "--iterations",
                "1",
                "--population",
                "3",
                "--output",
                "./kernup_results",
            ],
        )
        assert first.exit_code == 0

        lines = [line.strip() for line in first.output.splitlines()]
        run_id = next(line.split(":", maxsplit=1)[1].strip() for line in lines if line.startswith("Run ID:"))

        resumed = runner.invoke(
            cli,
            [
                "optimize",
                "--hf",
                "Qwen/Qwen2.5-7B",
                "--phase",
                "1",
                "--dry-run",
                "--allow-no-gpu",
                "--iterations",
                "1",
                "--population",
                "3",
                "--resume",
                "--output",
                "./kernup_results",
            ],
        )
        assert resumed.exit_code == 0
        assert f"Resuming run: {run_id}" in resumed.output

        runs = [p for p in Path("kernup_results").glob("run_*") if p.is_dir()]
        assert len(runs) == 1
        db_path = Path("kernup_results") / run_id / "kernup.db"

        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute("SELECT notes FROM results").fetchall()

        max_gen = -1
        for row in rows:
            notes = str(row[0]) if row and row[0] else ""
            match = re.search(r"gen=(\d+)", notes)
            if match:
                max_gen = max(max_gen, int(match.group(1)))

        assert max_gen >= 3


def test_bench_real_mode_prints_measured_metrics(monkeypatch) -> None:
    runner = CliRunner()

    def _fake_benchmark(**_kwargs):
        return RuntimeBenchmarkResult(tok_s=77.7, latency_ms=11.2, ttft_ms=88.1, vram_used_gb=6.4)

    monkeypatch.setattr("kernup.cli.bench.benchmark_hf_model", _fake_benchmark)

    result = runner.invoke(
        cli,
        [
            "bench",
            "--hf",
            "Qwen/Qwen2.5-7B",
            "--real",
        ],
    )

    assert result.exit_code == 0
    assert "Real benchmark mode enabled." in result.output
    assert "Best tok/s: 77.700" in result.output


def test_patch_and_bench_block_model_mismatch() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        run_dir = Path("kernup_results") / "run_20260306_000004_abcd12"
        run_dir.mkdir(parents=True)
        db_path = run_dir / "kernup.db"

        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE runs (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL DEFAULT '',
                    timestamp TEXT,
                    generation INTEGER,
                    block_size INTEGER,
                    num_warps INTEGER,
                    num_stages INTEGER,
                    kv_strategy TEXT,
                    split_k INTEGER,
                    mutation_type TEXT
                );
                CREATE TABLE results (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    tok_s REAL,
                    ttft_ms REAL,
                    latency_ms REAL,
                    vram_used_gb REAL,
                    is_best INTEGER,
                    notes TEXT
                );
                """
            )
            conn.execute(
                """
                INSERT INTO runs (id, model_id, timestamp, generation, block_size, num_warps, num_stages, kv_strategy, split_k, mutation_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run_20260306_000004_abcd12",
                    "Qwen/Qwen2.5-7B",
                    "2026-03-06T00:00:00+00:00",
                    0,
                    0,
                    0,
                    0,
                    "n/a",
                    0,
                    "dry-run",
                ),
            )
            conn.execute(
                """
                INSERT INTO results (id, run_id, tok_s, ttft_ms, latency_ms, vram_used_gb, is_best, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (str(uuid4()), "run_20260306_000004_abcd12", 22.2, 197.84, 93.38, 5.0, 1, "best"),
            )
            conn.commit()

        patch_result = runner.invoke(
            cli,
            [
                "patch",
                "--hf",
                "Qwen/Qwen3.5-4B",
                "--results",
                "./kernup_results",
                "--format",
                "simple",
                "--output",
                "./patch_out",
            ],
        )
        assert patch_result.exit_code != 0
        assert "Model mismatch" in patch_result.output

        bench_result = runner.invoke(
            cli,
            [
                "bench",
                "--hf",
                "Qwen/Qwen3.5-4B",
                "--results",
                "./kernup_results",
            ],
        )
        assert bench_result.exit_code != 0
        assert "Model mismatch" in bench_result.output


def test_optimize_resume_blocks_model_mismatch() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        run_dir = Path("kernup_results") / "run_20260306_000005_qwer12"
        run_dir.mkdir(parents=True)
        db_path = run_dir / "kernup.db"

        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE runs (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL DEFAULT '',
                    timestamp TEXT,
                    generation INTEGER,
                    block_size INTEGER,
                    num_warps INTEGER,
                    num_stages INTEGER,
                    kv_strategy TEXT,
                    split_k INTEGER,
                    mutation_type TEXT
                );
                CREATE TABLE results (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    tok_s REAL,
                    ttft_ms REAL,
                    latency_ms REAL,
                    vram_used_gb REAL,
                    is_best INTEGER,
                    notes TEXT
                );
                """
            )
            conn.execute(
                """
                INSERT INTO runs (id, model_id, timestamp, generation, block_size, num_warps, num_stages, kv_strategy, split_k, mutation_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run_20260306_000005_qwer12",
                    "Qwen/Qwen2.5-7B",
                    "2026-03-06T00:00:00+00:00",
                    0,
                    0,
                    0,
                    0,
                    "n/a",
                    0,
                    "dry-run",
                ),
            )
            conn.commit()

        result = runner.invoke(
            cli,
            [
                "optimize",
                "--hf",
                "Qwen/Qwen3.5-4B",
                "--phase",
                "1",
                "--dry-run",
                "--allow-no-gpu",
                "--resume",
                "--output",
                "./kernup_results",
            ],
        )

        assert result.exit_code != 0
        assert "Resume model mismatch" in result.output
