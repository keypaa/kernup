"""Kernel generation interface for phase 2."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class GenerationRequest:
    reference_kernel: str
    previous_best_kernel: str
    mutation_type: str
    generation: int


class KernelGenerator:
    """Generates candidate kernels for phase 2 evolution."""

    def __init__(self, dry_run: bool, seed: int = 123) -> None:
        self.dry_run = dry_run
        self._rng = random.Random(seed)

    def generate(self, request: GenerationRequest) -> str:
        if not self.dry_run:
            raise NotImplementedError("Real local LLM generation is not implemented yet")

        variant_id = self._rng.randint(10, 999)
        if request.mutation_type == "light":
            body = "return x"
        elif request.mutation_type == "medium":
            body = "y = x\n    return y"
        else:
            body = "y = x\n    z = y\n    return z"

        return (
            "import triton\n\n"
            "@triton.jit\n"
            f"def kernel_{request.generation}_{variant_id}(x):\n"
            f"    {body}\n"
        )
