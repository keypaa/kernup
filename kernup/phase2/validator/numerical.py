"""Numerical correctness validation for generated kernels."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose


@dataclass(frozen=True)
class NumericalValidationResult:
    ok: bool
    max_abs_diff: float
    details: str | None = None


def validate_numerical(
    reference: list[float],
    candidate: list[float],
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> NumericalValidationResult:
    """Validate numerical closeness between candidate and reference outputs."""
    if len(reference) != len(candidate):
        return NumericalValidationResult(
            ok=False,
            max_abs_diff=float("inf"),
            details="Output lengths differ",
        )

    max_diff = 0.0
    for ref, got in zip(reference, candidate, strict=True):
        diff = abs(ref - got)
        max_diff = max(max_diff, diff)
        if not isclose(ref, got, rel_tol=rtol, abs_tol=atol):
            return NumericalValidationResult(
                ok=False,
                max_abs_diff=max_diff,
                details=f"Mismatch ref={ref} got={got}",
            )

    return NumericalValidationResult(ok=True, max_abs_diff=max_diff)
