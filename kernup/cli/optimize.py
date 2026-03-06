"""Optimize command entrypoint (phase 1 dry-run implementation)."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

import click

from kernup.errors import UserError
from kernup.phase1.search import run_phase1_search
from kernup.phase2.pipeline import run_phase2_validation_pipeline
from kernup.storage.db import ResultRecord, RunRecord, create_schema, insert_result, insert_run, open_connection
from kernup.utils.gpu import ensure_gpu_available
from kernup.utils.runs import create_run_artifacts


@click.command("optimize")
@click.option("--hf", "hf_model", required=True, help="HuggingFace model id.")
@click.option("--phase", type=click.Choice(["1", "2"], case_sensitive=False), default="1", show_default=True)
@click.option(
    "--target",
    type=click.Choice(["throughput", "latency", "balanced"], case_sensitive=False),
    default="balanced",
    show_default=True,
)
@click.option("--iterations", default=20, show_default=True, type=int)
@click.option("--population", default=4, show_default=True, type=int)
@click.option("--plateau-window", default=10, show_default=True, type=int)
@click.option("--plateau-threshold", default=0.01, show_default=True, type=float)
@click.option("--output", default="./kernup_results", show_default=True)
@click.option("--dry-run", is_flag=True, default=False, help="Run synthetic phase 1 without GPU benchmarking.")
@click.option(
    "--allow-no-gpu",
    is_flag=True,
    default=False,
    help="Bypass GPU requirement for local setup and dry-run validation.",
)
def optimize_command(
    hf_model: str,
    phase: str,
    target: str,
    iterations: int,
    population: int,
    plateau_window: int,
    plateau_threshold: float,
    output: str,
    dry_run: bool,
    allow_no_gpu: bool,
) -> None:
    """Optimize a model using phase 1 search (dry-run supported)."""
    if not dry_run:
        raise click.ClickException(
            "Full optimization is not implemented yet. Use --dry-run during bootstrap."
        )

    try:
        gpu_status = ensure_gpu_available(allow_no_gpu=allow_no_gpu)
    except UserError as exc:
        raise click.ClickException(str(exc)) from exc

    artifacts = create_run_artifacts(output)
    now_iso = datetime.now().astimezone().isoformat()
    cache_dir = artifacts.run_dir / ".triton_cache"

    click.echo(f"Model: {hf_model}")
    click.echo(f"Target: {target}")
    click.echo(gpu_status.reason)

    if phase == "2":
        click.echo("Running phase 2 dry-run validation pipeline...")
        candidate_kernel = "def kernel(x):\n    return x\n"
        result = run_phase2_validation_pipeline(
            kernel_code=candidate_kernel,
            target=target,
            dry_run=True,
            max_healing_attempts=3,
        )

        with open_connection(artifacts.db_path) as conn:
            create_schema(conn)
            run_record = RunRecord(
                id=artifacts.run_id,
                timestamp=now_iso,
                generation=0,
                block_size=0,
                num_warps=0,
                num_stages=0,
                kv_strategy="n/a",
                split_k=0,
                mutation_type="phase2",
            )
            insert_run(conn, run_record)

            if result.benchmark is None:
                tok_s = 0.0
                latency_ms = 0.0
                ttft_ms = 0.0
                vram_used_gb = 0.0
                notes = "phase2_failed_before_benchmark"
            else:
                tok_s = result.benchmark.tok_s
                latency_ms = result.benchmark.latency_ms
                ttft_ms = result.benchmark.ttft_ms
                vram_used_gb = result.benchmark.vram_used_gb
                notes = f"phase2_dry_run_healed_attempts={result.healed_attempts}"

            insert_result(
                conn,
                ResultRecord(
                    id=str(uuid4()),
                    run_id=artifacts.run_id,
                    tok_s=tok_s,
                    ttft_ms=ttft_ms,
                    latency_ms=latency_ms,
                    vram_used_gb=vram_used_gb,
                    is_best=1,
                    notes=notes,
                ),
            )

        artifacts.log_path.write_text(
            "KERNUP optimize phase2 dry-run log\n"
            f"timestamp={now_iso}\n"
            f"run_id={artifacts.run_id}\n"
            f"model={hf_model}\n"
            f"target={target}\n"
            f"static_ok={result.static.ok}\n"
            f"numerical_ok={result.numerical.ok if result.numerical else False}\n"
            f"benchmark_ok={result.benchmark.ok if result.benchmark else False}\n"
            f"healed_attempts={result.healed_attempts}\n",
            encoding="utf-8",
        )

        phase2_path = artifacts.run_dir / "phase2_validation.json"
        phase2_payload = {
            "static_ok": result.static.ok,
            "static_error": result.static.error_message,
            "numerical_ok": result.numerical.ok if result.numerical else False,
            "benchmark": {
                "tok_s": result.benchmark.tok_s if result.benchmark else 0.0,
                "latency_ms": result.benchmark.latency_ms if result.benchmark else 0.0,
                "ttft_ms": result.benchmark.ttft_ms if result.benchmark else 0.0,
                "vram_used_gb": result.benchmark.vram_used_gb if result.benchmark else 0.0,
            },
            "healed_attempts": result.healed_attempts,
        }
        phase2_path.write_text(json.dumps(phase2_payload, indent=2), encoding="utf-8")

        click.echo(f"Run ID: {artifacts.run_id}")
        click.echo(f"Run dir: {artifacts.run_dir}")
        click.echo(f"Static gate: {result.static.ok}")
        click.echo(f"Numerical gate: {result.numerical.ok if result.numerical else False}")
        click.echo(f"Benchmark gate: {result.benchmark.ok if result.benchmark else False}")
        click.echo(f"Validation artifact: {phase2_path}")
        click.echo("Phase 2 dry-run complete.")
        return

    click.echo("Running phase 1 dry-run search...")

    result = run_phase1_search(
        iterations=iterations,
        population=population,
        plateau_window=plateau_window,
        plateau_threshold=plateau_threshold,
        target=target,
        cache_dir=cache_dir,
        gpu_compute_capability="dry-run",
        dry_run=True,
    )

    with open_connection(artifacts.db_path) as conn:
        create_schema(conn)
        run_record = RunRecord(
            id=artifacts.run_id,
            timestamp=now_iso,
            generation=len(result.history_best_tok_s) - 1,
            block_size=result.best_config.block_size,
            num_warps=result.best_config.num_warps,
            num_stages=result.best_config.num_stages,
            kv_strategy=result.best_config.kv_strategy,
            split_k=result.best_config.split_k,
            mutation_type="phase1",
        )
        insert_run(conn, run_record)

        for score in result.evaluations:
            insert_result(
                conn,
                ResultRecord(
                    id=str(uuid4()),
                    run_id=artifacts.run_id,
                    tok_s=score.tok_s,
                    ttft_ms=score.ttft_ms,
                    latency_ms=score.latency_ms,
                    vram_used_gb=score.vram_used_gb,
                    is_best=1 if score == result.best else 0,
                    notes=f"gen={score.generation}",
                ),
            )

    artifacts.log_path.write_text(
        "KERNUP optimize dry-run log\n"
        f"timestamp={now_iso}\n"
        f"run_id={artifacts.run_id}\n"
        f"model={hf_model}\n"
        f"target={target}\n"
        f"iterations={iterations}\n"
        f"population={population}\n"
        f"stopped_on_plateau={result.stopped_on_plateau}\n",
        encoding="utf-8",
    )

    progression_path = artifacts.run_dir / "phase1_progression.json"
    progression_path.write_text(
        json.dumps({"best_tok_s_by_generation": result.history_best_tok_s}, indent=2),
        encoding="utf-8",
    )

    click.echo(f"Run ID: {artifacts.run_id}")
    click.echo(f"Run dir: {artifacts.run_dir}")
    click.echo(f"Best tok/s: {result.best.tok_s}")
    click.echo(f"Best config: {result.best.config}")
    click.echo(f"Progression: {progression_path}")
    click.echo("Phase 1 dry-run complete.")
