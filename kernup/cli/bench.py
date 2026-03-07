"""Bench command for comparing baseline and optimized results."""

from __future__ import annotations

from pathlib import Path
import json
import sqlite3

import click

from kernup.benchmark.runtime import benchmark_hf_model
from kernup.utils.runs import latest_run_dir


def _latest_run_dir(results_dir: Path) -> Path:
    run_dir = latest_run_dir(results_dir, require_db=True)
    if run_dir is None:
        raise click.ClickException(f"No run folders found under {results_dir}")
    return run_dir


def _best_row(db_path: Path) -> tuple[float, float, float, float]:
    if not db_path.exists():
        raise click.ClickException(f"Missing database at {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            """
            SELECT tok_s, ttft_ms, latency_ms, vram_used_gb
            FROM results
            ORDER BY is_best DESC, tok_s DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise click.ClickException("No results found in selected run database")
    return float(row[0]), float(row[1]), float(row[2]), float(row[3])


def _run_model_id(db_path: Path, run_id: str) -> str | None:
    if not db_path.exists():
        raise click.ClickException(f"Missing database at {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        try:
            row = conn.execute(
                """
                SELECT model_id
                FROM runs
                WHERE id = ?
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    finally:
        conn.close()

    if row is None:
        return None
    return str(row[0]) if row[0] is not None else None


@click.command("bench")
@click.option("--hf", "hf_model", required=True, help="HuggingFace model id.")
@click.option("--results", "results_dir", default="./kernup_results", show_default=True)
@click.option("--seq-lens", default="128,512,2048", show_default=True)
@click.option("--batch-sizes", default="1,4,8,16", show_default=True)
@click.option("--real", "real_run", is_flag=True, default=False, help="Run real GPU benchmark now.")
@click.option(
    "--prompt",
    "prompt_text",
    default="Write a short summary of GPU kernel optimization best practices.",
    show_default=True,
    help="Benchmark prompt used for real GPU timing.",
)
@click.option("--max-new-tokens", default=32, show_default=True, type=int)
@click.option("--warmup-runs", default=1, show_default=True, type=int)
@click.option("--measure-runs", default=2, show_default=True, type=int)
@click.option("--export", is_flag=True, default=False, help="Export benchmark summary as JSON.")
@click.option("--output", "output_dir", default="./kernup_results", show_default=True)
@click.option(
    "--allow-model-mismatch",
    is_flag=True,
    default=False,
    help="Allow benchmark summary even when the run model id differs from --hf.",
)
def bench_command(
    hf_model: str,
    results_dir: str,
    seq_lens: str,
    batch_sizes: str,
    real_run: bool,
    prompt_text: str,
    max_new_tokens: int,
    warmup_runs: int,
    measure_runs: int,
    export: bool,
    output_dir: str,
    allow_model_mismatch: bool,
) -> None:
    """Show benchmark-style summary for the latest run."""
    if max_new_tokens <= 0:
        raise click.ClickException("--max-new-tokens must be greater than 0")
    if warmup_runs < 0:
        raise click.ClickException("--warmup-runs must be >= 0")
    if measure_runs <= 0:
        raise click.ClickException("--measure-runs must be greater than 0")

    if real_run:
        try:
            measured = benchmark_hf_model(
                hf_model=hf_model,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                warmup_runs=warmup_runs,
                measure_runs=measure_runs,
            )
        except Exception as exc:
            raise click.ClickException(f"Real benchmark failed: {exc}") from exc

        tok_s = measured.tok_s
        ttft_ms = measured.ttft_ms
        latency_ms = measured.latency_ms
        vram_used = measured.vram_used_gb
        run_id = "live"
        click.echo("Real benchmark mode enabled.")
    else:
        run_dir = _latest_run_dir(Path(results_dir))
        run_id = run_dir.name
        db_path = run_dir / "kernup.db"
        tok_s, ttft_ms, latency_ms, vram_used = _best_row(db_path)
        run_model = _run_model_id(db_path, run_id)

        if run_model is None:
            if not allow_model_mismatch:
                raise click.ClickException(
                    "Run model metadata is missing. Re-run profile/optimize with current schema, "
                    "or pass --allow-model-mismatch to bypass this check."
                )
            click.echo("Warning: run model metadata missing, bypass enabled.")
        elif run_model != hf_model:
            if not allow_model_mismatch:
                raise click.ClickException(
                    f"Model mismatch: run uses '{run_model}' but --hf is '{hf_model}'. "
                    "Pass --allow-model-mismatch to force bench summary."
                )
            click.echo(f"Warning: model mismatch bypass enabled ({run_model} vs {hf_model}).")

    # Baseline is placeholder until real profile baseline is persisted per run.
    baseline_tok_s = 0.0
    speedup = 0.0 if baseline_tok_s == 0 else tok_s / baseline_tok_s

    click.echo(f"Model: {hf_model}")
    click.echo(f"Run: {run_id}")
    click.echo(f"Seq lens: {seq_lens}")
    click.echo(f"Batch sizes: {batch_sizes}")
    click.echo(f"Best tok/s: {tok_s:.3f}")
    click.echo(f"TTFT ms: {ttft_ms:.3f}")
    click.echo(f"Latency ms: {latency_ms:.3f}")
    click.echo(f"VRAM GB: {vram_used:.3f}")
    click.echo(f"Speedup vs baseline: {speedup:.3f}x")

    if export:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / f"bench_{run_id}.json"
        summary_path.write_text(
            json.dumps(
                {
                    "model": hf_model,
                    "run_id": run_id,
                    "seq_lens": seq_lens,
                    "batch_sizes": batch_sizes,
                    "best": {
                        "tok_s": tok_s,
                        "ttft_ms": ttft_ms,
                        "latency_ms": latency_ms,
                        "vram_used_gb": vram_used,
                    },
                    "baseline_tok_s": baseline_tok_s,
                    "speedup": speedup,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        click.echo(f"Export: {summary_path}")
