"""Benchmark validation stage for generated kernels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkValidationResult:
    ok: bool
    tok_s: float
    latency_ms: float
    ttft_ms: float
    vram_used_gb: float


def benchmark_kernel(kernel_code: str, target: str, dry_run: bool) -> BenchmarkValidationResult:
    """Benchmark the candidate kernel or return synthetic metrics in dry-run."""
    if not dry_run:
        raise NotImplementedError("Real phase 2 benchmarking is not implemented yet")

    complexity = max(1, len(kernel_code.splitlines()))
    tok_s = round(18.0 + min(40.0, complexity * 0.6), 3)
    latency_ms = round(max(8.0, 140.0 - tok_s * 2.1), 3)
    ttft_ms = round(max(15.0, 260.0 - tok_s * 2.8), 3)
    vram_used_gb = 5.0

    if target == "latency":
        latency_ms = round(latency_ms * 0.88, 3)
    elif target == "throughput":
        tok_s = round(tok_s * 1.08, 3)

    return BenchmarkValidationResult(
        ok=True,
        tok_s=tok_s,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        vram_used_gb=vram_used_gb,
    )
