"""Profile command (dry-run first vertical slice)."""

from __future__ import annotations

import click

from kernup.config.schema import AppConfig
from kernup.errors import UserError
from kernup.utils.gpu import ensure_gpu_available


@click.command("profile")
@click.option("--hf", "hf_model", required=True, help="HuggingFace model id.")
@click.option("--device", default=None, help="Device id (default: cuda:0).")
@click.option("--dry-run", is_flag=True, default=False, help="Validate setup only.")
@click.option(
    "--allow-no-gpu",
    is_flag=True,
    default=False,
    help="Bypass GPU requirement for local setup and dry-run validation.",
)
def profile_command(hf_model: str, device: str | None, dry_run: bool, allow_no_gpu: bool) -> None:
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
        click.echo("Dry-run complete: configuration and environment checks passed.")
        return

    raise click.ClickException(
        "Full profiling is not implemented yet. Use --dry-run during bootstrap."
    )
