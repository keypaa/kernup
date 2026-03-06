"""Status and clean utility commands."""

from __future__ import annotations

from pathlib import Path
import shutil
import sqlite3
import time

import click


def _run_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted([p for p in results_root.iterdir() if p.is_dir() and p.name.startswith("run_")])


@click.command("status")
@click.option("--results", "results_dir", default="./kernup_results", show_default=True)
def status_command(results_dir: str) -> None:
    """Show discovered run folders and basic best score summary."""
    root = Path(results_dir)
    runs = _run_dirs(root)

    if not runs:
        click.echo(f"No runs found under {root}")
        return

    click.echo(f"Found {len(runs)} run(s) under {root}")
    for run_dir in runs:
        db_path = run_dir / "kernup.db"
        if not db_path.exists():
            click.echo(f"- {run_dir.name}: missing kernup.db")
            continue

        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro&immutable=1", uri=True)
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    SELECT tok_s, latency_ms, ttft_ms
                    FROM results
                    ORDER BY is_best DESC, tok_s DESC
                    LIMIT 1
                    """
                )
                row = cursor.fetchone()
            finally:
                cursor.close()
        finally:
            conn.close()

        if row is None:
            click.echo(f"- {run_dir.name}: no result rows")
            continue

        click.echo(
            f"- {run_dir.name}: best tok/s={row[0]:.3f}, latency_ms={row[1]:.3f}, ttft_ms={row[2]:.3f}"
        )


@click.command("clean")
@click.option("--results", "results_dir", default="./kernup_results", show_default=True)
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
def clean_command(results_dir: str, yes: bool) -> None:
    """Delete generated run folders under the results directory."""
    root = Path(results_dir)
    runs = _run_dirs(root)

    if not runs:
        click.echo(f"No runs to clean under {root}")
        return

    if not yes:
        confirmed = click.confirm(f"Delete {len(runs)} run(s) under {root}?", default=False)
        if not confirmed:
            click.echo("Aborted.")
            return

    for run_dir in runs:
        last_exc: OSError | None = None
        for _ in range(10):
            try:
                shutil.rmtree(run_dir)
                last_exc = None
                break
            except OSError as exc:
                last_exc = exc
                time.sleep(0.1)
        if last_exc is not None:
            raise click.ClickException(f"Failed to remove {run_dir}: {last_exc}") from last_exc

    click.echo(f"Removed {len(runs)} run(s) from {root}")
