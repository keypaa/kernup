from pathlib import Path

from kernup.benchmark.runtime import RuntimeBenchmarkResult
from kernup.phase1.search import run_phase1_search


def test_phase1_max_mode_runs_refinement_calls(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def _fake_score_config(*args, **kwargs):
        calls.append(kwargs)
        cfg = args[0]
        generation = int(kwargs["generation"])
        measure_runs = int(kwargs["measure_runs"])
        tok_s = float(cfg.block_size / 32.0 + measure_runs)
        return RuntimeBenchmarkResult(  # type: ignore[return-value]
            tok_s=tok_s,
            latency_ms=max(1.0, 200.0 - tok_s),
            ttft_ms=max(1.0, 300.0 - tok_s),
            vram_used_gb=6.0,
        )

    class _Score:
        def __init__(self, tok_s, latency_ms, ttft_ms, vram_used_gb, generation, config):
            self.tok_s = tok_s
            self.latency_ms = latency_ms
            self.ttft_ms = ttft_ms
            self.vram_used_gb = vram_used_gb
            self.generation = generation
            self.config = config

    def _fake_score_config_kernel(*args, **kwargs):
        result = _fake_score_config(*args, **kwargs)
        cfg = args[0]
        return _Score(
            tok_s=result.tok_s,
            latency_ms=result.latency_ms,
            ttft_ms=result.ttft_ms,
            vram_used_gb=result.vram_used_gb,
            generation=int(kwargs["generation"]),
            config=cfg.as_dict(),
        )

    monkeypatch.setattr("kernup.phase1.search.score_config", _fake_score_config_kernel)

    result = run_phase1_search(
        iterations=1,
        population=4,
        plateau_window=3,
        plateau_threshold=0.01,
        target="balanced",
        cache_dir=tmp_path,
        gpu_compute_capability="sm_75",
        dry_run=False,
        hf_model="Qwen/Qwen2.5-7B",
        search_mode="max",
        max_refine_top_k=2,
        max_refine_warmup_runs=2,
        max_refine_measure_runs=5,
    )

    assert len(result.evaluations) > 8
    assert any(int(call["measure_runs"]) == 5 for call in calls)


def test_phase1_max_mode_allows_restart_on_plateau(monkeypatch, tmp_path: Path) -> None:
    def _flat_score(*args, **kwargs):
        cfg = args[0]

        class _Score:
            tok_s = 10.0
            latency_ms = 100.0
            ttft_ms = 200.0
            vram_used_gb = 6.0
            generation = int(kwargs["generation"])
            config = cfg.as_dict()

        return _Score()

    monkeypatch.setattr("kernup.phase1.search.score_config", _flat_score)

    result = run_phase1_search(
        iterations=3,
        population=4,
        plateau_window=1,
        plateau_threshold=0.5,
        target="balanced",
        cache_dir=tmp_path,
        gpu_compute_capability="sm_75",
        dry_run=True,
        search_mode="max",
        max_restarts=1,
    )

    assert result.stopped_on_plateau
    assert len(result.history_best_tok_s) >= 2
