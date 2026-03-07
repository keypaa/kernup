"""Real runtime benchmarking based on HuggingFace generation latency."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


@dataclass(frozen=True)
class HFModelBundle:
    torch: Any
    tokenizer: Any
    model: Any
    device: str


@dataclass(frozen=True)
class RuntimeBenchmarkResult:
    tok_s: float
    latency_ms: float
    ttft_ms: float
    vram_used_gb: float


_MODEL_CACHE: dict[str, HFModelBundle] = {}


def get_hf_model_bundle(hf_model: str) -> HFModelBundle:
    """Load and cache tokenizer/model bundle for repeated benchmarking calls."""
    cached = _MODEL_CACHE.get(hf_model)
    if cached is not None:
        return cached

    torch = __import__("torch")
    transformers = __import__("transformers")
    AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")
    AutoTokenizer = getattr(transformers, "AutoTokenizer")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real benchmarking")

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    bundle = HFModelBundle(torch=torch, tokenizer=tokenizer, model=model, device=device)
    _MODEL_CACHE[hf_model] = bundle
    return bundle


def benchmark_hf_model(
    hf_model: str,
    prompt_text: str,
    max_new_tokens: int,
    warmup_runs: int,
    measure_runs: int,
) -> RuntimeBenchmarkResult:
    """Run a simple generation benchmark and return measured runtime metrics."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")
    if measure_runs <= 0:
        raise ValueError("measure_runs must be > 0")

    bundle = get_hf_model_bundle(hf_model)
    torch = bundle.torch
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device

    encoded = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in encoded.items()}

    def _sync() -> None:
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            _sync()

        ttft_samples_ms: list[float] = []
        latency_samples_ms: list[float] = []
        tok_s_samples: list[float] = []
        vram_samples_gb: list[float] = []

        for _ in range(measure_runs):
            torch.cuda.reset_peak_memory_stats()

            ttft_start = time.perf_counter()
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            _sync()
            ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

            start = time.perf_counter()
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            _sync()
            elapsed_s = time.perf_counter() - start

            generated_tokens = int(outputs.shape[-1] - inputs["input_ids"].shape[-1])
            generated_tokens = max(1, generated_tokens)
            tok_s = generated_tokens / max(elapsed_s, 1e-9)
            latency_ms = (elapsed_s * 1000.0) / generated_tokens
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024.0**3)

            ttft_samples_ms.append(ttft_ms)
            latency_samples_ms.append(latency_ms)
            tok_s_samples.append(tok_s)
            vram_samples_gb.append(peak_vram_gb)

    return RuntimeBenchmarkResult(
        tok_s=round(sum(tok_s_samples) / len(tok_s_samples), 3),
        latency_ms=round(sum(latency_samples_ms) / len(latency_samples_ms), 3),
        ttft_ms=round(sum(ttft_samples_ms) / len(ttft_samples_ms), 3),
        vram_used_gb=round(sum(vram_samples_gb) / len(vram_samples_gb), 3),
    )
