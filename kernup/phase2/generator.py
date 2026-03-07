"""Kernel generation interface for phase 2."""

from __future__ import annotations

from dataclasses import dataclass
import random
import re

from kernup.benchmark import get_hf_model_bundle
from kernup.phase2.prompt import PromptBudget, build_generation_prompt


@dataclass(frozen=True)
class GenerationRequest:
    reference_kernel: str
    previous_best_kernel: str
    mutation_type: str
    generation: int


class KernelGenerator:
    """Generates candidate kernels for phase 2 evolution."""

    def __init__(
        self,
        dry_run: bool,
        seed: int = 123,
        hf_model: str = "",
        prompt_text: str = "",
        generation_max_new_tokens: int = 128,
    ) -> None:
        self.dry_run = dry_run
        self._rng = random.Random(seed)
        self._hf_model = hf_model
        self._prompt_text = prompt_text
        self._generation_max_new_tokens = generation_max_new_tokens

    def _fallback_kernel(self, request: GenerationRequest) -> str:
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

    def _extract_kernel_candidate(self, text: str) -> str:
        matches = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
        candidate = matches[0] if matches else text
        candidate = candidate.strip()
        if "import triton" not in candidate or "@triton.jit" not in candidate:
            raise ValueError("Generated candidate is missing required Triton markers")
        return candidate

    def _generate_with_model(self, request: GenerationRequest) -> str:
        if not self._hf_model:
            raise ValueError("hf_model is required for real generation")

        bundle = get_hf_model_bundle(self._hf_model)
        torch = bundle.torch
        tokenizer = bundle.tokenizer
        model = bundle.model

        prompt = build_generation_prompt(
            constraints=(
                "Return only Python code for a Triton kernel candidate. "
                "Must include import triton and @triton.jit."
            ),
            gpu_profile="cuda",
            model_arch=self._hf_model,
            best_phase1=f"mutation_type={request.mutation_type}; generation={request.generation}",
            reference_kernel=request.reference_kernel,
            previous_best_kernel=request.previous_best_kernel,
            final_instruction=(
                f"Produce one candidate kernel variation. {self._prompt_text}".strip()
            ),
            budget=PromptBudget(max_chars=5000),
        )

        encoded = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(bundle.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self._generation_max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        return self._extract_kernel_candidate(text)

    def generate(self, request: GenerationRequest) -> str:
        if self.dry_run:
            return self._fallback_kernel(request)

        try:
            return self._generate_with_model(request)
        except Exception:
            # Keep phase 2 progressing even if text generation quality varies.
            return self._fallback_kernel(request)
