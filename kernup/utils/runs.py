"""Run artifact directory and identifier helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import secrets


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    db_path: Path
    log_path: Path
    profile_path: Path


def generate_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now().astimezone()
    ts = current.strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(3)
    return f"run_{ts}_{suffix}"


def create_run_artifacts(output_root: str | Path, run_id: str | None = None) -> RunArtifacts:
    rid = run_id or generate_run_id()
    root = Path(output_root)
    run_dir = root / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    return RunArtifacts(
        run_id=rid,
        run_dir=run_dir,
        db_path=run_dir / "kernup.db",
        log_path=run_dir / "kernup.log",
        profile_path=run_dir / "profile.json",
    )
