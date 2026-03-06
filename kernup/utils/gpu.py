"""GPU detection helpers with optional bypass for CPU-only development."""

from __future__ import annotations

from dataclasses import dataclass

from kernup.errors import UserError


@dataclass(frozen=True)
class GPUCheckResult:
    available: bool
    bypassed: bool
    reason: str


def ensure_gpu_available(allow_no_gpu: bool = False) -> GPUCheckResult:
    """Validate CUDA availability unless bypass is explicitly enabled."""
    if allow_no_gpu:
        return GPUCheckResult(
            available=False,
            bypassed=True,
            reason="Bypassing GPU check (--allow-no-gpu).",
        )

    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency may be absent locally
        raise UserError(
            "PyTorch is required to check GPU availability. "
            "Install torch or use --allow-no-gpu for dry-run setup checks."
        ) from exc

    if not torch.cuda.is_available():
        raise UserError(
            "No NVIDIA GPU detected. Use --allow-no-gpu for setup validation "
            "or run on an NVIDIA-enabled machine for optimization."
        )

    return GPUCheckResult(available=True, bypassed=False, reason="CUDA GPU detected.")
