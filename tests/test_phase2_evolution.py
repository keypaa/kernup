from kernup.phase2.evolution import run_phase2_evolution
from kernup.phase2.generator import GenerationRequest, KernelGenerator


def test_generator_dry_run_produces_triton_code() -> None:
    generator = KernelGenerator(dry_run=True, seed=7)
    code = generator.generate(
        GenerationRequest(
            reference_kernel="import triton\n\n@triton.jit\ndef seed(x):\n    return x\n",
            previous_best_kernel="import triton\n\n@triton.jit\ndef best(x):\n    return x\n",
            mutation_type="light",
            generation=1,
        )
    )
    assert "import triton" in code
    assert "@triton.jit" in code


def test_phase2_evolution_dry_run_returns_best_and_history() -> None:
    result = run_phase2_evolution(
        iterations=4,
        population=4,
        plateau_window=3,
        plateau_threshold=0.01,
        target="balanced",
        dry_run=True,
        max_healing_attempts=3,
        seed=11,
    )
    assert len(result.evaluations) >= 4
    assert len(result.history_best_tok_s) >= 1
    assert result.best.pipeline.benchmark is not None
    assert result.best.pipeline.benchmark.tok_s > 0


def test_generator_real_mode_falls_back_when_model_generation_fails(monkeypatch) -> None:
    generator = KernelGenerator(
        dry_run=False,
        seed=7,
        hf_model="Qwen/Qwen2.5-7B",
        prompt_text="test",
    )

    def _fail(_request):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(generator, "_generate_with_model", _fail)

    code = generator.generate(
        GenerationRequest(
            reference_kernel="import triton\n\n@triton.jit\ndef seed(x):\n    return x\n",
            previous_best_kernel="import triton\n\n@triton.jit\ndef best(x):\n    return x\n",
            mutation_type="medium",
            generation=2,
        )
    )
    assert "import triton" in code
    assert "@triton.jit" in code
