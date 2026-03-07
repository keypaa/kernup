"""Patch command for generating deployment integration snippets."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import click

from kernup.patch import render_sglang_patch, render_simple_patch, render_tgi_patch, render_vllm_patch, smoke_check_patch
from kernup.utils.runs import latest_run_dir


def _latest_run_dir(results_dir: Path) -> Path:
    run_dir = latest_run_dir(results_dir, require_db=True)
    if run_dir is None:
        raise click.ClickException(f"No run folders found under {results_dir}")
    return run_dir


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
@click.option(
    "--allow-model-mismatch",
    is_flag=True,
    default=False,
    help="Allow patch generation even when the run model id differs from --hf.",
)
@click.option("--smoke", is_flag=True, default=False, help="Run patch import/apply smoke check.")
@click.option(
    "--smoke-with-model",
    is_flag=True,
    default=False,
    help="Use a real HF model for smoke check (simple format only, GPU recommended).",
)
@click.option(
    "--smoke-prompt",
    default="Say hello in one sentence.",
    show_default=True,
    help="Prompt used for model-backed patch smoke checks.",
)
@click.option("--smoke-max-new-tokens", default=8, show_default=True, type=int)
def patch_command(
    hf_model: str,
    results_dir: str,
    patch_format: str,
    output_dir: str,
    allow_model_mismatch: bool,
    smoke: bool,
    smoke_with_model: bool,
    smoke_prompt: str,
    smoke_max_new_tokens: int,
) -> None:
    """Generate a patch artifact from the latest available run."""
    results_root = Path(results_dir)
    run_dir = _latest_run_dir(results_root)
    run_id = run_dir.name
    db_path = run_dir / "kernup.db"
    tok_s = _best_tok_s(db_path)
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
                "Pass --allow-model-mismatch to force patch generation."
            )
        click.echo(f"Warning: model mismatch bypass enabled ({run_model} vs {hf_model}).")

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

    if smoke_max_new_tokens <= 0:
        raise click.ClickException("--smoke-max-new-tokens must be greater than 0")

    smoke_message = None
    if smoke:
        if smoke_with_model and patch_format != "simple":
            raise click.ClickException("--smoke-with-model currently supports only --format simple")
        try:
            smoke_message = smoke_check_patch(
                file_path=file_path,
                patch_format=patch_format,
                hf_model=hf_model,
                with_model=smoke_with_model,
                prompt_text=smoke_prompt,
                max_new_tokens=smoke_max_new_tokens,
            )
        except Exception as exc:
            raise click.ClickException(f"Patch smoke check failed: {exc}") from exc

    click.echo(f"Using run: {run_id}")
    click.echo(f"Best tok/s: {tok_s:.3f}")
    click.echo(f"Patch file: {file_path}")
    if smoke_message:
        click.echo(f"Smoke: {smoke_message}")
