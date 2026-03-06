"""Typed config schema used by CLI commands."""

from __future__ import annotations

from dataclasses import dataclass

from kernup.config import defaults


@dataclass(frozen=True)
class AppConfig:
    """Minimal runtime config for the current implementation stage."""

    device: str = defaults.DEFAULT_DEVICE
    allow_no_gpu: bool = defaults.DEFAULT_ALLOW_NO_GPU

    @classmethod
    def from_cli(cls, device: str | None, allow_no_gpu: bool) -> "AppConfig":
        selected_device = device or defaults.DEFAULT_DEVICE
        return cls(device=selected_device, allow_no_gpu=allow_no_gpu)
