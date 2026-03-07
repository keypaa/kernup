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
        dtype=torch.float16,
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
    batch_size: int = 1,
    use_cache: bool = True,
    pad_to_multiple_of: int | None = None,
) -> RuntimeBenchmarkResult:
    """Run a simple generation benchmark and return measured runtime metrics."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")
    if measure_runs <= 0:
        raise ValueError("measure_runs must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
        raise ValueError("pad_to_multiple_of must be > 0 when provided")

    bundle = get_hf_model_bundle(hf_model)
    torch = bundle.torch
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = bundle.device

    prompts: str | list[str] = prompt_text
    if batch_size > 1:
        prompts = [prompt_text] * batch_size

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True if batch_size > 1 or pad_to_multiple_of is not None else False,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    inputs = {k: v.to(device) for k, v in encoded.items()}
    input_lengths = None
    if "attention_mask" in inputs:
        input_lengths = inputs["attention_mask"].sum(dim=1)

    def _sync() -> None:
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
                pad_token_id=tokenizer.pad_token_id,
            )
            _sync()

        ttft_samples_ms: list[float] = []
        latency_samples_ms: list[float] = []
        tok_s_samples: list[float] = []
        vram_samples_gb: list[float] = []

        for _ in range(measure_runs):
            torch.cuda.reset_peak_memory_stats()

            ttft_start = time.perf_counter()
            _ = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                use_cache=use_cache,
                pad_token_id=tokenizer.pad_token_id,
            )
            _sync()
            ttft_ms = (time.perf_counter() - ttft_start) * 1000.0

            start = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
                pad_token_id=tokenizer.pad_token_id,
            )
            _sync()
            elapsed_s = time.perf_counter() - start

            if input_lengths is None:
                generated_tokens = int(outputs.shape[-1] - inputs["input_ids"].shape[-1])
            else:
                generated_tokens = int(
                    sum(max(1, int(outputs.shape[-1] - int(length.item()))) for length in input_lengths)
                )
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
