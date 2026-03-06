"""Patch command for generating deployment integration snippets."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import click

from kernup.patch import render_sglang_patch, render_simple_patch, render_tgi_patch, render_vllm_patch


def _latest_run_dir(results_dir: Path) -> Path:
    runs = sorted([p for p in results_dir.glob("run_*") if p.is_dir()])
    if not runs:
        raise click.ClickException(f"No run folders found under {results_dir}")
    return runs[-1]


def _best_tok_s(db_path: Path) -> float:
    if not db_path.exists():
        raise click.ClickException(f"Missing database at {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            """
            SELECT tok_s
            FROM results
            ORDER BY is_best DESC, tok_s DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise click.ClickException("No results found in selected run database")
    return float(row[0])


@click.command("patch")
@click.option("--hf", "hf_model", required=True, help="HuggingFace model id.")
@click.option("--results", "results_dir", default="./kernup_results", show_default=True)
@click.option(
    "--format",
    "patch_format",
    type=click.Choice(["simple", "vllm", "tgi", "sglang"], case_sensitive=False),
    default="simple",
    show_default=True,
)
@click.option("--output", "output_dir", default="./patch", show_default=True)
def patch_command(hf_model: str, results_dir: str, patch_format: str, output_dir: str) -> None:
    """Generate a patch artifact from the latest available run."""
    results_root = Path(results_dir)
    run_dir = _latest_run_dir(results_root)
    run_id = run_dir.name
    tok_s = _best_tok_s(run_dir / "kernup.db")

    renderers = {
        "simple": render_simple_patch,
        "vllm": render_vllm_patch,
        "tgi": render_tgi_patch,
        "sglang": render_sglang_patch,
    }
    content = renderers[patch_format](hf_model=hf_model, run_id=run_id, best_tok_s=tok_s)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    file_path = output_root / f"patch_{patch_format}.py"
    file_path.write_text(content, encoding="utf-8")

    click.echo(f"Using run: {run_id}")
    click.echo(f"Best tok/s: {tok_s:.3f}")
    click.echo(f"Patch file: {file_path}")
