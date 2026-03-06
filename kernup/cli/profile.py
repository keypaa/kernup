"""Profile command (dry-run first vertical slice)."""

from __future__ import annotations

import json
from datetime import datetime
from dataclasses import asdict
from uuid import uuid4

import click

from kernup.config.schema import AppConfig
from kernup.errors import UserError
from kernup.profiler import detect_hardware, run_baseline, run_breakdown
from kernup.storage.db import ResultRecord, RunRecord, create_schema, insert_result, insert_run, open_connection
from kernup.storage.export import export_results_json
from kernup.utils.gpu import ensure_gpu_available
from kernup.utils.runs import create_run_artifacts


@click.command("profile")
@click.option("--hf", "hf_model", required=True, help="HuggingFace model id.")
@click.option("--device", default=None, help="Device id (default: cuda:0).")
@click.option("--dry-run", is_flag=True, default=False, help="Validate setup only.")
@click.option("--export", is_flag=True, default=False, help="Export run snapshot as JSON.")
@click.option("--output", default="./kernup_results", show_default=True, help="Result root folder.")
@click.option(
    "--allow-no-gpu",
    is_flag=True,
    default=False,
    help="Bypass GPU requirement for local setup and dry-run validation.",
)
def profile_command(
    hf_model: str,
    device: str | None,
    dry_run: bool,
    export: bool,
    output: str,
    allow_no_gpu: bool,
) -> None:
    """Run setup validation and (later) profiling for a model."""
    cfg = AppConfig.from_cli(device=device, allow_no_gpu=allow_no_gpu)

    try:
        gpu_status = ensure_gpu_available(allow_no_gpu=cfg.allow_no_gpu)
    except UserError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Model: {hf_model}")
    click.echo(f"Device: {cfg.device}")
    click.echo(gpu_status.reason)

    if dry_run:
        gpu_profile = detect_hardware(gpu_status)
        baseline_summary = run_baseline(dry_run=True)
        breakdown_summary = run_breakdown(dry_run=True)

        artifacts = create_run_artifacts(output)
        now_iso = datetime.now().astimezone().isoformat()
        run_record = RunRecord(
            id=artifacts.run_id,
            timestamp=now_iso,
            generation=0,
            block_size=0,
            num_warps=0,
            num_stages=0,
            kv_strategy="n/a",
            split_k=0,
            mutation_type="dry-run",
        )
        result_record = ResultRecord(
            id=str(uuid4()),
            run_id=artifacts.run_id,
            tok_s=0.0,
            ttft_ms=0.0,
            latency_ms=0.0,
            vram_used_gb=0.0,
            is_best=1,
            notes=f"Dry-run setup validation for model {hf_model}",
        )

        with open_connection(artifacts.db_path) as conn:
            create_schema(conn)
            insert_run(conn, run_record)
            insert_result(conn, result_record)
            if export:
                export_results_json(conn, artifacts.run_id, artifacts.run_dir / "results_export.json")

        artifacts.log_path.write_text(
            "KERNUP dry-run log\n"
            f"timestamp={now_iso}\n"
            f"run_id={artifacts.run_id}\n"
            f"model={hf_model}\n"
            f"device={cfg.device}\n"
            f"allow_no_gpu={cfg.allow_no_gpu}\n",
            encoding="utf-8",
        )

        profile_payload = {
            "run_id": artifacts.run_id,
            "timestamp": now_iso,
            "model": hf_model,
            "device": cfg.device,
            "dry_run": dry_run,
            "gpu": asdict(gpu_profile),
            "baseline": baseline_summary.to_dict(),
            "breakdown": {"bottlenecks": breakdown_summary.bottlenecks},
        }
        artifacts.profile_path.write_text(json.dumps(profile_payload, indent=2), encoding="utf-8")

        click.echo(f"Run ID: {artifacts.run_id}")
        click.echo(f"Run dir: {artifacts.run_dir}")
        click.echo(f"Database: {artifacts.db_path}")
        click.echo(f"Profile: {artifacts.profile_path}")
        if export:
            click.echo(f"Export: {artifacts.run_dir / 'results_export.json'}")
        click.echo("Dry-run complete: configuration and environment checks passed.")
        return

    raise click.ClickException(
        "Full profiling is not implemented yet. Use --dry-run during bootstrap."
    )
