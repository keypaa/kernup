"""Typed data interfaces shared across Kernup modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUProfile:
    name: str
    sm_count: int
    sram_per_sm_kb: int
    hbm_bandwidth_tbs: float
    vram_total_gb: float
    vram_free_gb: float
    compute_capability: str


@dataclass(frozen=True)
class BaselineResult:
    tok_s: float
    ttft_ms: float
    latency_ms: float
    vram_used_gb: float
    seq_len: int


@dataclass(frozen=True)
class KernelScore:
    tok_s: float
    ttft_ms: float
    latency_ms: float
    vram_used_gb: float
    generation: int
    config: dict[str, object]


@dataclass(frozen=True)
class KernelVariant:
    code: str
    parent_id: str
    mutation_type: str
    generation: int
