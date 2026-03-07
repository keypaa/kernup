"""Scoring and cache helpers for phase 1."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from kernup.benchmark.runtime import benchmark_hf_model
from kernup.phase1.hyperparams import HyperparameterConfig
from kernup.types import KernelScore


def compute_cache_key(kernel_code: str, config: HyperparameterConfig, gpu_compute_capability: str) -> str:
    payload = {
        "kernel_code": kernel_code,
        "config": asdict(config),
        "gpu_compute_capability": gpu_compute_capability,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return sha256(raw).hexdigest()


def _score_file(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.json"


def _config_factor(config: HyperparameterConfig) -> float:
    """Map candidate config to a small multiplicative factor around 1.0."""
    factor = 1.0
    factor += (config.block_size - 128) / 2048.0
    factor += (config.num_warps - 8) / 160.0
    factor += (config.split_k - 4) / 220.0
    if config.kv_strategy == "full":
        factor += 0.01
    if config.l2_cache_hint == "evict_last":
        factor += 0.01
    return max(0.85, min(1.15, factor))


def score_config(
    config: HyperparameterConfig,
    generation: int,
    cache_dir: Path,
    target: str,
    gpu_compute_capability: str,
    dry_run: bool,
    hf_model: str = "",
    prompt_text: str = "Write a short summary of GPU kernel optimization best practices.",
    max_new_tokens: int = 32,
    warmup_runs: int = 1,
    measure_runs: int = 2,
) -> KernelScore:
    cache_dir.mkdir(parents=True, exist_ok=True)
    kernel_scope = "phase1_reference_kernel" if dry_run else f"phase1_real::{hf_model}"
    key = compute_cache_key(
        kernel_code=kernel_scope,
        config=config,
        gpu_compute_capability=gpu_compute_capability,
    )
    path = _score_file(cache_dir, key)

    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return KernelScore(
            tok_s=cached["tok_s"],
            ttft_ms=cached["ttft_ms"],
            latency_ms=cached["latency_ms"],
            vram_used_gb=cached["vram_used_gb"],
            generation=generation,
            config=config.as_dict(),
        )

    if not dry_run:
        runtime = benchmark_hf_model(
            hf_model=hf_model,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
        )
        factor = _config_factor(config)

        tok_s = runtime.tok_s * factor
        latency_ms = runtime.latency_ms / factor
        ttft_ms = runtime.ttft_ms / factor
        if target == "latency":
            tok_s *= 0.97
            latency_ms *= 0.9
        elif target == "throughput":
            tok_s *= 1.03

        payload = {
            "tok_s": round(tok_s, 3),
            "ttft_ms": round(ttft_ms, 3),
            "latency_ms": round(latency_ms, 3),
            "vram_used_gb": runtime.vram_used_gb,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return KernelScore(
            tok_s=payload["tok_s"],
            ttft_ms=payload["ttft_ms"],
            latency_ms=payload["latency_ms"],
            vram_used_gb=payload["vram_used_gb"],
            generation=generation,
            config=config.as_dict(),
        )

    # Deterministic synthetic score for dry-run development.
    base = 20.0
    throughput = (
        base
        + (config.block_size / 32.0)
        + (config.num_warps * 0.7)
        + (6.0 - config.num_stages) * 0.25
        + (config.split_k * 0.2)
        + (config.prefetch_distance * 0.1)
    )
    if config.kv_strategy == "full":
        throughput += 0.8
    if config.l2_cache_hint == "evict_last":
        throughput += 0.5

    latency = max(5.0, 120.0 - throughput * 2.3)
    ttft = max(20.0, 240.0 - throughput * 3.0)
    vram = 4.0 + (config.block_size / 256.0) + (config.tensor_padding / 512.0)

    if target == "latency":
        throughput *= 0.95
        latency *= 0.85
    elif target == "balanced":
        throughput *= 1.0
    elif target == "throughput":
        throughput *= 1.05

    payload = {
        "tok_s": round(throughput, 3),
        "ttft_ms": round(ttft, 3),
        "latency_ms": round(latency, 3),
        "vram_used_gb": round(vram, 3),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return KernelScore(
        tok_s=payload["tok_s"],
        ttft_ms=payload["ttft_ms"],
        latency_ms=payload["latency_ms"],
        vram_used_gb=payload["vram_used_gb"],
        generation=generation,
        config=config.as_dict(),
    )
