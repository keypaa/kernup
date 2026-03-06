from click.testing import CliRunner
from pathlib import Path
from uuid import uuid4

from kernup.cli.main import cli
from kernup.storage.db import ResultRecord, RunRecord, create_schema, insert_result, insert_run, list_results_for_run, open_connection


def test_cli_help_shows_profile_command() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "profile" in result.output


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
