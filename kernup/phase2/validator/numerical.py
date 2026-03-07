"""Numerical correctness validation for generated kernels."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from kernup.benchmark import get_hf_model_bundle


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


def validate_model_numerical(
    hf_model: str,
    prompt_text: str,
    max_new_tokens: int,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> NumericalValidationResult:
    """Validate deterministic model outputs by comparing repeated forward passes."""
    if max_new_tokens <= 0:
        return NumericalValidationResult(ok=False, max_abs_diff=float("inf"), details="max_new_tokens must be > 0")

    try:
        bundle = get_hf_model_bundle(hf_model)
        torch = bundle.torch
        tokenizer = bundle.tokenizer
        model = bundle.model
        device = bundle.device

        encoded = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            out_a = model(**inputs)
            out_b = model(**inputs)
            logits_a = out_a.logits
            logits_b = out_b.logits

            max_abs_diff = float((logits_a - logits_b).abs().max().item())

            scale = max(1.0, float(logits_a.abs().max().item()))
            tolerance = atol + (rtol * scale)
            if max_abs_diff > tolerance:
                return NumericalValidationResult(
                    ok=False,
                    max_abs_diff=max_abs_diff,
                    details=f"Forward pass drift exceeded tolerance ({max_abs_diff:.3e} > {tolerance:.3e})",
                )

            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_tokens = int(generated.shape[-1] - inputs["input_ids"].shape[-1])
            if generated_tokens <= 0:
                return NumericalValidationResult(
                    ok=False,
                    max_abs_diff=max_abs_diff,
                    details="Generation produced zero new tokens",
                )

            return NumericalValidationResult(
                ok=True,
                max_abs_diff=max_abs_diff,
                details=f"generated_tokens={generated_tokens}",
            )
    except Exception as exc:
        return NumericalValidationResult(ok=False, max_abs_diff=float("inf"), details=str(exc))
