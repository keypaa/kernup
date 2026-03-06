"""Prompt assembly with lightweight context-budget control."""

from __future__ import annotations

from dataclasses import dataclass

from kernup.errors import KernupError


@dataclass(frozen=True)
class PromptBudget:
    max_chars: int = 12000


def _require_reference_kernel(reference_kernel: str) -> None:
    if not reference_kernel.strip():
        raise KernupError(
            "Phase 2 prompt generation blocked: reference kernel is missing. "
            "Provide a non-empty reference kernel before invoking the generator."
        )


def _clamp(text: str, max_chars: int) -> str:
    if max_chars <= 3:
        return "..."[:max_chars]
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_generation_prompt(
    constraints: str,
    gpu_profile: str,
    model_arch: str,
    best_phase1: str,
    reference_kernel: str,
    previous_best_kernel: str,
    final_instruction: str,
    budget: PromptBudget | None = None,
) -> str:
    """Build a structured English prompt while respecting character budget."""
    _require_reference_kernel(reference_kernel)
    cfg = budget or PromptBudget()

    blocks = [
        "[Role and Constraints]\n" + constraints,
        "[GPU Profile]\n" + gpu_profile,
        "[Model Architecture]\n" + model_arch,
        "[Best Phase 1]\n" + best_phase1,
        "[Reference Kernel]\n" + reference_kernel,
        "[Previous Best Kernel]\n" + previous_best_kernel,
        "[Instruction]\n" + final_instruction,
    ]

    joined = "\n\n".join(blocks)
    if len(joined) <= cfg.max_chars:
        return joined

    # Always preserve the instruction section, then distribute remaining budget.
    instruction_block = "[Instruction]\n" + _clamp(final_instruction, min(800, cfg.max_chars // 3))

    head = "\n\n".join(blocks[:4])
    protected = len(instruction_block) + 4
    remaining = max(40, cfg.max_chars - protected)
    half = remaining // 2
    ref = _clamp(reference_kernel, max(10, half - 20))
    prev = _clamp(previous_best_kernel, max(10, remaining - half - 30))
    middle = "\n\n".join(
        [
            _clamp(head, max(20, remaining // 2)),
            "[Reference Kernel]\n" + ref,
            "[Previous Best Kernel]\n" + prev,
        ]
    )

    candidate = middle + "\n\n" + instruction_block
    if len(candidate) <= cfg.max_chars:
        return candidate

    # Final fallback keeps the instruction and the two kernel blocks.
    fallback = "\n\n".join(
        [
            "[Reference Kernel]\n" + _clamp(reference_kernel, max(10, cfg.max_chars // 4)),
            "[Previous Best Kernel]\n" + _clamp(previous_best_kernel, max(10, cfg.max_chars // 4)),
            instruction_block,
        ]
    )
    return _clamp(fallback, cfg.max_chars)
