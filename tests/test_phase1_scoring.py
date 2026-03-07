from pathlib import Path

from kernup.phase1.hyperparams import HyperparameterConfig
from kernup.phase1.scoring import score_config


def _cfg(num_warps: int, kv_strategy: str, tensor_padding: int) -> HyperparameterConfig:
    return HyperparameterConfig(
        block_size=128,
        num_warps=num_warps,
        num_stages=3,
        kv_strategy=kv_strategy,
        tensor_padding=tensor_padding,
        split_k=2,
        prefetch_distance=2,
        l2_cache_hint="none",
    )


def test_score_config_real_passes_runtime_tuning_knobs(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def _fake_bench(**kwargs):
        calls.append(kwargs)
        from kernup.benchmark.runtime import RuntimeBenchmarkResult

        return RuntimeBenchmarkResult(tok_s=12.0, latency_ms=80.0, ttft_ms=220.0, vram_used_gb=6.0)

    monkeypatch.setattr("kernup.phase1.scoring.benchmark_hf_model", _fake_bench)

    result = score_config(
        config=_cfg(num_warps=16, kv_strategy="chunked", tensor_padding=256),
        generation=0,
        cache_dir=tmp_path,
        target="balanced",
        gpu_compute_capability="sm_75",
        dry_run=False,
        hf_model="Qwen/Qwen2.5-7B",
        prompt_text="hello",
        max_new_tokens=16,
        warmup_runs=1,
        measure_runs=2,
    )

    assert result.tok_s == 12.0
    assert len(calls) == 1
    assert calls[0]["batch_size"] == 4
    assert calls[0]["use_cache"] is False
    assert calls[0]["pad_to_multiple_of"] == 256


def test_score_config_real_uses_cache_hit(monkeypatch, tmp_path: Path) -> None:
    call_count = 0

    def _fake_bench(**kwargs):
        nonlocal call_count
        call_count += 1
        from kernup.benchmark.runtime import RuntimeBenchmarkResult

        return RuntimeBenchmarkResult(tok_s=10.0 + call_count, latency_ms=70.0, ttft_ms=200.0, vram_used_gb=5.5)

    monkeypatch.setattr("kernup.phase1.scoring.benchmark_hf_model", _fake_bench)

    config = _cfg(num_warps=4, kv_strategy="full", tensor_padding=128)

    first = score_config(
        config=config,
        generation=1,
        cache_dir=tmp_path,
        target="balanced",
        gpu_compute_capability="sm_75",
        dry_run=False,
        hf_model="Qwen/Qwen2.5-7B",
    )
    second = score_config(
        config=config,
        generation=2,
        cache_dir=tmp_path,
        target="balanced",
        gpu_compute_capability="sm_75",
        dry_run=False,
        hf_model="Qwen/Qwen2.5-7B",
    )

    assert call_count == 1
    assert first.tok_s == second.tok_s
