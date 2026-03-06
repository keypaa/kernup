"""Hardware detection helpers for profiling flow."""

from __future__ import annotations

from kernup.types import GPUProfile
from kernup.utils.gpu import GPUCheckResult


def detect_hardware(gpu_status: GPUCheckResult) -> GPUProfile:
    """Return detected GPU profile or a dry-run placeholder profile."""
    if gpu_status.bypassed or not gpu_status.available:
        return GPUProfile(
            name="NO_GPU_DRY_RUN",
            sm_count=0,
            sram_per_sm_kb=0,
            hbm_bandwidth_tbs=0.0,
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            compute_capability="n/a",
        )

    # At this stage, CUDA availability is already validated by ensure_gpu_available.
    import torch  # pragma: no cover - requires GPU runtime

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return GPUProfile(
        name=props.name,
        sm_count=int(props.multi_processor_count),
        sram_per_sm_kb=0,
        hbm_bandwidth_tbs=0.0,
        vram_total_gb=round(total_bytes / (1024**3), 3),
        vram_free_gb=round(free_bytes / (1024**3), 3),
        compute_capability=f"{props.major}.{props.minor}",
    )
