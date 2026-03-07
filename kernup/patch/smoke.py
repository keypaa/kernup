"""Patch artifact smoke checks."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from kernup.benchmark import get_hf_model_bundle


class _DummyModel:
    pass


def _load_module(file_path: Path):
    spec = spec_from_file_location("kernup_generated_patch", file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load patch module from {file_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoke_check_patch(
    file_path: Path,
    patch_format: str,
    hf_model: str,
    with_model: bool,
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    """Run a syntax/import/apply smoke check for generated patch artifacts."""
    module = _load_module(file_path)

    if patch_format == "simple":
        if not hasattr(module, "apply_kernup_patch"):
            raise RuntimeError("Simple patch does not expose apply_kernup_patch")

        apply_fn = getattr(module, "apply_kernup_patch")
        model = _DummyModel()

        if with_model:
            bundle = get_hf_model_bundle(hf_model)
            torch = bundle.torch
            tokenizer = bundle.tokenizer
            model = bundle.model
            encoded = tokenizer(prompt_text, return_tensors="pt")
            inputs = {k: v.to(bundle.device) for k, v in encoded.items()}

            patched = apply_fn(model)
            metadata = getattr(patched, "kernup_metadata", None)
            if not isinstance(metadata, dict):
                raise RuntimeError("Simple patch did not attach kernup_metadata")
            with torch.no_grad():
                _ = patched.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            return "Simple patch imported, applied to model, and generated successfully."

        patched = apply_fn(model)
        metadata = getattr(patched, "kernup_metadata", None)
        if not isinstance(metadata, dict):
            raise RuntimeError("Simple patch did not attach kernup_metadata")
        return "Simple patch imported and applied successfully."

    symbol_map = {
        "vllm": "KERNUP_VLLM_PLUGIN",
        "tgi": "KERNUP_TGI_PLUGIN",
        "sglang": "KERNUP_SGLANG_PLUGIN",
    }
    symbol = symbol_map.get(patch_format)
    if symbol is None:
        raise RuntimeError(f"Unsupported patch format for smoke check: {patch_format}")

    payload = getattr(module, symbol, None)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected {symbol} to be a dict")
    if payload.get("hf_model") != hf_model:
        raise RuntimeError(f"Patch model mismatch in payload: {payload.get('hf_model')} != {hf_model}")
    return f"{patch_format} patch imported and payload validated successfully."
