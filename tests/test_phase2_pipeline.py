import pytest

from kernup.errors import KernupError
from kernup.phase2.healer import heal_with_retries
from kernup.phase2.pipeline import run_phase2_validation_pipeline
from kernup.phase2.prompt import PromptBudget, build_generation_prompt
from kernup.phase2.validator.benchmark import benchmark_kernel
from kernup.phase2.validator.numerical import validate_numerical
from kernup.phase2.validator.static import validate_static


def test_static_validator_rejects_missing_triton() -> None:
    result = validate_static("def kernel(x):\n    return x\n")
    assert not result.ok
    assert result.error_message is not None


def test_static_validator_accepts_triton_jit_pattern() -> None:
    code = "import triton\n\n@triton.jit\ndef kernel(x):\n    return x\n"
    result = validate_static(code)
    assert result.ok


def test_numerical_validator_detects_mismatch() -> None:
    result = validate_numerical([1.0, 2.0], [1.0, 5.0], rtol=1e-3, atol=1e-3)
    assert not result.ok


def test_benchmark_dry_run_returns_metrics() -> None:
    code = "import triton\n\n@triton.jit\ndef kernel(x):\n    return x\n"
    result = benchmark_kernel(code, target="balanced", dry_run=True)
    assert result.ok
    assert result.tok_s > 0


def test_prompt_builder_respects_budget() -> None:
    prompt = build_generation_prompt(
        constraints="A" * 200,
        gpu_profile="B" * 200,
        model_arch="C" * 200,
        best_phase1="D" * 200,
        reference_kernel="E" * 4000,
        previous_best_kernel="F" * 4000,
        final_instruction="G" * 200,
        budget=PromptBudget(max_chars=1800),
    )
    assert len(prompt) <= 1800
    assert "[Instruction]" in prompt


def test_prompt_builder_raises_when_reference_kernel_missing() -> None:
    with pytest.raises(KernupError):
        build_generation_prompt(
            constraints="constraints",
            gpu_profile="gpu",
            model_arch="model",
            best_phase1="phase1",
            reference_kernel="   ",
            previous_best_kernel="prev",
            final_instruction="instruction",
            budget=PromptBudget(max_chars=1200),
        )


def test_healer_stops_after_success() -> None:
    def validator(code: str) -> tuple[bool, str | None]:
        if "@triton.jit" in code:
            return True, None
        return False, "Missing Triton JIT decoration"

    def fixer(code: str, _err: str) -> str:
        return "import triton\n\n@triton.jit\ndef kernel(x):\n    return x\n"

    attempts = heal_with_retries("def kernel(x):\n    return x\n", validator, fixer, max_attempts=3)
    assert attempts[-1].success
    assert len(attempts) == 2


def test_phase2_pipeline_heals_and_benchmarks() -> None:
    result = run_phase2_validation_pipeline(
        kernel_code="def kernel(x):\n    return x\n",
        target="balanced",
        dry_run=True,
        max_healing_attempts=3,
    )
    assert result.static.ok
    assert result.numerical is not None and result.numerical.ok
    assert result.benchmark is not None and result.benchmark.ok
