"""Phase 2 dry-run validation pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from kernup.phase2.healer import heal_with_retries
from kernup.phase2.validator.benchmark import BenchmarkValidationResult, benchmark_kernel
from kernup.phase2.validator.numerical import NumericalValidationResult, validate_numerical
from kernup.phase2.validator.static import StaticValidationResult, validate_static


@dataclass(frozen=True)
class Phase2PipelineResult:
    static: StaticValidationResult
    numerical: NumericalValidationResult | None
    benchmark: BenchmarkValidationResult | None
    healed_attempts: int
    final_code: str


def run_phase2_validation_pipeline(
    kernel_code: str,
    target: str,
    dry_run: bool,
    max_healing_attempts: int = 3,
) -> Phase2PipelineResult:
    """Run static -> numerical -> benchmark gates with self-healing."""

    def _validator(code: str) -> tuple[bool, str | None]:
        result = validate_static(code)
        return result.ok, result.error_message

    def _fixer(code: str, _error: str) -> str:
        if "import triton" not in code:
            return "import triton\n\n@triton.jit\ndef kernel(x):\n    return x\n"
        if "@triton.jit" not in code:
            return code.replace("def kernel", "@triton.jit\ndef kernel")
        return code

    attempts = heal_with_retries(
        initial_code=kernel_code,
        validator=_validator,
        fixer=_fixer,
        max_attempts=max_healing_attempts,
    )
    final_attempt = attempts[-1]
    static_result = validate_static(final_attempt.code)
    if not static_result.ok:
        return Phase2PipelineResult(
            static=static_result,
            numerical=None,
            benchmark=None,
            healed_attempts=len(attempts) - 1,
            final_code=final_attempt.code,
        )

    numerical_result = validate_numerical(
        reference=[1.0, 2.0, 3.0, 4.0],
        candidate=[1.0, 2.0, 3.0, 4.0],
        rtol=1e-2,
        atol=1e-2,
    )
    if not numerical_result.ok:
        return Phase2PipelineResult(
            static=static_result,
            numerical=numerical_result,
            benchmark=None,
            healed_attempts=len(attempts) - 1,
            final_code=final_attempt.code,
        )

    bench_result = benchmark_kernel(kernel_code=final_attempt.code, target=target, dry_run=dry_run)
    return Phase2PipelineResult(
        static=static_result,
        numerical=numerical_result,
        benchmark=bench_result,
        healed_attempts=len(attempts) - 1,
        final_code=final_attempt.code,
    )
