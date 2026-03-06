"""Static validation for generated kernel code."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class StaticValidationResult:
    ok: bool
    error_message: str | None = None


def validate_static(kernel_code: str) -> StaticValidationResult:
    """Validate basic Python syntax and expected Triton markers."""
    if not kernel_code.strip():
        return StaticValidationResult(ok=False, error_message="Kernel code is empty")

    try:
        ast.parse(kernel_code)
    except SyntaxError as exc:
        return StaticValidationResult(ok=False, error_message=f"SyntaxError: {exc.msg}")

    if "triton" not in kernel_code.lower():
        return StaticValidationResult(
            ok=False,
            error_message="Kernel code does not reference Triton",
        )

    if "@triton.jit" not in kernel_code and "triton.jit" not in kernel_code:
        return StaticValidationResult(
            ok=False,
            error_message="Missing Triton JIT decoration",
        )

    return StaticValidationResult(ok=True)
