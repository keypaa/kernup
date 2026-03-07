"""Optimize command entrypoint."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

import click

from kernup.errors import UserError
from kernup.phase1.search import run_phase1_search
from kernup.phase2.evolution import run_phase2_evolution
from kernup.storage.db import ResultRecord, RunRecord, create_schema, insert_result, open_connection, upsert_run
from kernup.utils.gpu import ensure_gpu_available
from kernup.utils.runs import create_run_artifacts, latest_run_dir


def _read_run_model(db_path: Path, run_id: str) -> str | None:
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
            row = None
    finally:
        conn.close()

    if row is None:
        return None
    return str(row[0]) if row[0] else None


def _last_generation(db_path: Path, phase: str) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT notes
            FROM results
            ORDER BY rowid ASC
            """
        ).fetchall()
    finally:
        conn.close()

    pattern = r"phase2_gen=(\d+)" if phase == "2" else r"(?<!phase2_)gen=(\d+)"
    maximum = -1
    for row in rows:
        notes = str(row[0]) if row and row[0] else ""
        match = re.search(pattern, notes)
        if match:
            maximum = max(maximum, int(match.group(1)))
    return maximum


def _merge_history(path: Path, key: str, new_history: list[float]) -> list[float]:
    if not path.exists():
        return new_history
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        previous = list(existing.get(key, []))
    except Exception:
        return new_history
    if not previous:
        return new_history
    # Drop first point because it corresponds to current starting generation.
    return previous + new_history[1:]


def _append_or_write_log(path: Path, content: str, append: bool) -> None:
    if append and path.exists():
        path.write_text(path.read_text(encoding="utf-8") + "\n" + content, encoding="utf-8")
        return
    path.write_text(content, encoding="utf-8")


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
@click.option("--resume", is_flag=True, default=False, help="Resume from the latest run metadata.")
@click.option("--dry-run", is_flag=True, default=False, help="Run synthetic phase 1 without GPU benchmarking.")
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
@click.option(
    "--search-mode",
    type=click.Choice(["standard", "max"], case_sensitive=False),
    default="standard",
    show_default=True,
    help="Phase 1 search strategy. 'max' uses multi-fidelity refinement and restart-on-stagnation.",
)
@click.option("--max-refine-top-k", default=2, show_default=True, type=int)
@click.option("--max-refine-warmup-runs", default=2, show_default=True, type=int)
@click.option("--max-refine-measure-runs", default=6, show_default=True, type=int)
@click.option("--max-stability-penalty", default=0.15, show_default=True, type=float)
@click.option("--max-restarts", default=1, show_default=True, type=int)
@click.option(
    "--allow-no-gpu",
    is_flag=True,
    default=False,
    help="Bypass GPU requirement for local setup and dry-run validation.",
)
@click.option(
    "--allow-model-mismatch",
    is_flag=True,
    default=False,
    help="Allow resume when latest run model differs from --hf.",
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
    resume: bool,
    dry_run: bool,
    prompt_text: str,
    max_new_tokens: int,
    warmup_runs: int,
    measure_runs: int,
    search_mode: str,
    max_refine_top_k: int,
    max_refine_warmup_runs: int,
    max_refine_measure_runs: int,
    max_stability_penalty: float,
    max_restarts: int,
    allow_no_gpu: bool,
    allow_model_mismatch: bool,
) -> None:
    """Optimize a model using phase 1 or phase 2 workflows."""
    if max_new_tokens <= 0:
        raise click.ClickException("--max-new-tokens must be greater than 0")
    if warmup_runs < 0:
        raise click.ClickException("--warmup-runs must be >= 0")
    if measure_runs <= 0:
        raise click.ClickException("--measure-runs must be greater than 0")
    if max_refine_top_k <= 0:
        raise click.ClickException("--max-refine-top-k must be greater than 0")
    if max_refine_warmup_runs < 0:
        raise click.ClickException("--max-refine-warmup-runs must be >= 0")
    if max_refine_measure_runs <= 0:
        raise click.ClickException("--max-refine-measure-runs must be greater than 0")
    if max_stability_penalty < 0:
        raise click.ClickException("--max-stability-penalty must be >= 0")
    if max_restarts < 0:
        raise click.ClickException("--max-restarts must be >= 0")

    try:
        gpu_status = ensure_gpu_available(allow_no_gpu=allow_no_gpu)
    except UserError as exc:
        raise click.ClickException(str(exc)) from exc

    if not dry_run and gpu_status.bypassed:
        raise click.ClickException("Real optimization requires GPU; remove --allow-no-gpu.")

    results_root = Path(output)
    resume_dir: Path | None = None
    start_generation = 0

    if resume:
        latest = latest_run_dir(results_root, require_db=True)
        if latest is None:
            raise click.ClickException(
                f"--resume requested but no runs found under {results_root}"
            )
        resume_dir = latest
        db_path = latest / "kernup.db"
        if not db_path.exists():
            raise click.ClickException(
                f"--resume requested but latest run has no database: {db_path}"
            )

        latest_model = _read_run_model(db_path, latest.name)
        if latest_model is None:
            if not allow_model_mismatch:
                raise click.ClickException(
                    "Latest run model metadata is missing; cannot validate --resume consistency. "
                    "Use --allow-model-mismatch to bypass."
                )
            click.echo("Warning: resume model metadata missing, bypass enabled.")
        elif latest_model != hf_model:
            if not allow_model_mismatch:
                raise click.ClickException(
                    f"Resume model mismatch: latest run uses '{latest_model}' but --hf is '{hf_model}'. "
                    "Use --allow-model-mismatch to bypass."
                )
            click.echo(f"Warning: resume model mismatch bypass enabled ({latest_model} vs {hf_model}).")

        start_generation = _last_generation(db_path, phase=phase) + 1

    artifacts = create_run_artifacts(output, run_id=resume_dir.name if resume_dir else None)
    now_iso = datetime.now().astimezone().isoformat()
    cache_dir = results_root / ".phase1_cache"
    gpu_compute_capability = "dry-run"
    if not dry_run:
        import torch

        major, minor = torch.cuda.get_device_capability(0)
        gpu_compute_capability = f"sm_{major}{minor}"

    click.echo(f"Model: {hf_model}")
    click.echo(f"Target: {target}")
    if phase == "1":
        click.echo(f"Search mode: {search_mode}")
    click.echo(gpu_status.reason)
    if resume_dir is not None:
        click.echo(f"Resuming run: {artifacts.run_id} (start generation: {start_generation})")

    if phase == "2":
        click.echo("Running phase 2 dry-run evolution loop..." if dry_run else "Running phase 2 real evolution loop...")
        result = run_phase2_evolution(
            iterations=iterations,
            population=population,
            plateau_window=plateau_window,
            plateau_threshold=plateau_threshold,
            target=target,
            dry_run=dry_run,
            max_healing_attempts=3,
            hf_model=hf_model,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
            start_generation=start_generation,
        )

        with open_connection(artifacts.db_path) as conn:
            create_schema(conn)
            run_record = RunRecord(
                id=artifacts.run_id,
                model_id=hf_model,
                timestamp=now_iso,
                generation=max((evaluation.generation for evaluation in result.evaluations), default=start_generation),
                block_size=0,
                num_warps=0,
                num_stages=0,
                kv_strategy="n/a",
                split_k=0,
                mutation_type="phase2-evolution",
            )
            upsert_run(conn, run_record)

            for evaluation in result.evaluations:
                bench = evaluation.pipeline.benchmark
                insert_result(
                    conn,
                    ResultRecord(
                        id=str(uuid4()),
                        run_id=artifacts.run_id,
                        tok_s=bench.tok_s if bench else 0.0,
                        ttft_ms=bench.ttft_ms if bench else 0.0,
                        latency_ms=bench.latency_ms if bench else 0.0,
                        vram_used_gb=bench.vram_used_gb if bench else 0.0,
                        is_best=1 if evaluation == result.best else 0,
                        notes=(
                            f"phase2_gen={evaluation.generation};"
                            f"mutation={evaluation.mutation_type};"
                            f"heals={evaluation.pipeline.healed_attempts}"
                        ),
                    ),
                )

        mode_label = "dry-run" if dry_run else "real"
        log_prefix = "KERNUP optimize phase2 resumed" if resume_dir is not None else "KERNUP optimize phase2"
        log_content = (
            f"{log_prefix} {mode_label} evolution log\n"
            f"timestamp={now_iso}\n"
            f"run_id={artifacts.run_id}\n"
            f"model={hf_model}\n"
            f"target={target}\n"
            f"iterations={iterations}\n"
            f"population={population}\n"
            f"start_generation={start_generation}\n"
            f"stopped_on_plateau={result.stopped_on_plateau}\n"
        )
        _append_or_write_log(artifacts.log_path, log_content, append=resume_dir is not None)

        phase2_path = artifacts.run_dir / "phase2_evolution.json"
        best_bench = result.best.pipeline.benchmark
        history = result.history_best_tok_s
        if resume_dir is not None:
            history = _merge_history(phase2_path, "history_best_tok_s", history)

        phase2_payload = {
            "best": {
                "generation": result.best.generation,
                "mutation_type": result.best.mutation_type,
                "static_ok": result.best.pipeline.static.ok,
                "numerical_ok": result.best.pipeline.numerical.ok if result.best.pipeline.numerical else False,
                "benchmark": {
                    "tok_s": best_bench.tok_s if best_bench else 0.0,
                    "latency_ms": best_bench.latency_ms if best_bench else 0.0,
                    "ttft_ms": best_bench.ttft_ms if best_bench else 0.0,
                    "vram_used_gb": best_bench.vram_used_gb if best_bench else 0.0,
                },
            },
            "history_best_tok_s": history,
            "stopped_on_plateau": result.stopped_on_plateau,
            "total_evaluations": len(result.evaluations),
        }
        phase2_path.write_text(json.dumps(phase2_payload, indent=2), encoding="utf-8")

        click.echo(f"Run ID: {artifacts.run_id}")
        click.echo(f"Run dir: {artifacts.run_dir}")
        click.echo(f"Best tok/s: {best_bench.tok_s if best_bench else 0.0}")
        click.echo(f"Best generation: {result.best.generation}")
        click.echo(f"Total evaluations: {len(result.evaluations)}")
        click.echo(f"Evolution artifact: {phase2_path}")
        click.echo("Phase 2 dry-run complete." if dry_run else "Phase 2 real run complete.")
        return

    click.echo("Running phase 1 dry-run search..." if dry_run else "Running phase 1 real search...")

    result = run_phase1_search(
        iterations=iterations,
        population=population,
        plateau_window=plateau_window,
        plateau_threshold=plateau_threshold,
        target=target,
        cache_dir=cache_dir,
        gpu_compute_capability=gpu_compute_capability,
        dry_run=dry_run,
        hf_model=hf_model,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
        warmup_runs=warmup_runs,
        measure_runs=measure_runs,
        search_mode=search_mode,
        max_refine_top_k=max_refine_top_k,
        max_refine_warmup_runs=max_refine_warmup_runs,
        max_refine_measure_runs=max_refine_measure_runs,
        max_stability_penalty=max_stability_penalty,
        max_restarts=max_restarts,
        start_generation=start_generation,
    )

    with open_connection(artifacts.db_path) as conn:
        create_schema(conn)
        run_record = RunRecord(
            id=artifacts.run_id,
            model_id=hf_model,
            timestamp=now_iso,
            generation=max((score.generation for score in result.evaluations), default=start_generation),
            block_size=result.best_config.block_size,
            num_warps=result.best_config.num_warps,
            num_stages=result.best_config.num_stages,
            kv_strategy=result.best_config.kv_strategy,
            split_k=result.best_config.split_k,
            mutation_type="phase1",
        )
        upsert_run(conn, run_record)

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

    mode_label = "dry-run" if dry_run else "real"
    log_prefix = "KERNUP optimize resumed" if resume_dir is not None else "KERNUP optimize"
    log_content = (
        f"{log_prefix} {mode_label} log\n"
        f"timestamp={now_iso}\n"
        f"run_id={artifacts.run_id}\n"
        f"model={hf_model}\n"
        f"target={target}\n"
        f"iterations={iterations}\n"
        f"population={population}\n"
        f"start_generation={start_generation}\n"
        f"stopped_on_plateau={result.stopped_on_plateau}\n"
    )
    _append_or_write_log(artifacts.log_path, log_content, append=resume_dir is not None)

    progression_path = artifacts.run_dir / "phase1_progression.json"
    history = result.history_best_tok_s
    if resume_dir is not None:
        history = _merge_history(progression_path, "best_tok_s_by_generation", history)

    progression_path.write_text(
        json.dumps({"best_tok_s_by_generation": history}, indent=2),
        encoding="utf-8",
    )

    click.echo(f"Run ID: {artifacts.run_id}")
    click.echo(f"Run dir: {artifacts.run_dir}")
    click.echo(f"Best tok/s: {result.best.tok_s}")
    click.echo(f"Best config: {result.best.config}")
    click.echo(f"Progression: {progression_path}")
    click.echo("Phase 1 dry-run complete." if dry_run else "Phase 1 real run complete.")
