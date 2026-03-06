# Kernup

Automatic kernel optimization for HuggingFace transformer inference workloads.

Kernup is a Python CLI project focused on profiling and optimization workflows for PyTorch-based serving stacks. The current implementation provides a robust dry-run development mode that works on CPU-only machines, including Windows environments without local NVIDIA GPUs.

## Current Scope

Implemented commands:
- `kernup profile`
- `kernup optimize`
- `kernup patch`
- `kernup bench`
- `kernup status`
- `kernup clean`

Implemented architecture slices:
- Profiling dry-run with run artifact persistence
- Phase 1 dry-run genetic hyperparameter search with caching scaffolding
- Phase 2 dry-run validation, healing, prompt budget logic, generation and evolution loop
- Patch generation formats (`simple`, `vllm`, `tgi`, `sglang`)
- Bench summary and export from stored run results

## Requirements

- Python 3.11+
- Windows, Linux, or macOS
- Optional GPU stack for future non-dry-run execution:
  - NVIDIA GPU
  - PyTorch + CUDA
  - Triton

## Installation

### Conda (recommended on Windows)

```powershell
conda create -y -n kernup-dev python=3.11
conda activate kernup-dev
python -m pip install -e .[dev]
```

### Basic editable install

```bash
python -m pip install -e .
```

## Quickstart

Dry-run profile on CPU-only setup:

```powershell
kernup profile --hf Qwen/Qwen2.5-7B --dry-run --allow-no-gpu --output ./kernup_results --export
```

Dry-run optimize phase 1:

```powershell
kernup optimize --hf Qwen/Qwen2.5-7B --phase 1 --dry-run --allow-no-gpu --iterations 10 --population 4
```

Dry-run optimize phase 2:

```powershell
kernup optimize --hf Qwen/Qwen2.5-7B --phase 2 --dry-run --allow-no-gpu --iterations 6 --population 4
```

Generate patch output:

```powershell
kernup patch --hf Qwen/Qwen2.5-7B --results ./kernup_results --format simple --output ./patch_out
```

Run bench summary export:

```powershell
kernup bench --hf Qwen/Qwen2.5-7B --results ./kernup_results --export --output ./kernup_results
```

## CLI Reference

### profile

```text
kernup profile --hf <repo/model> [--device cuda:0] [--dry-run] [--allow-no-gpu] [--export] [--output ./kernup_results]
```

### optimize

```text
kernup optimize --hf <repo/model> --phase 1|2 [--target throughput|latency|balanced] [--iterations N] [--population N] [--plateau-window N] [--plateau-threshold F] [--dry-run] [--allow-no-gpu] [--output ./kernup_results]
```

### patch

```text
kernup patch --hf <repo/model> --results ./kernup_results --format simple|vllm|tgi|sglang --output ./patch
```

### bench

```text
kernup bench --hf <repo/model> --results ./kernup_results [--seq-lens 128,512,2048] [--batch-sizes 1,4,8,16] [--export] [--output ./kernup_results]
```

### utilities

```text
kernup status --results ./kernup_results
kernup clean --results ./kernup_results [--yes]
```

## Output Structure

Each run creates:

```text
kernup_results/
  run_YYYYMMDD_HHMMSS_xxxxxx/
    kernup.db
    kernup.log
    profile.json            # profile command
    results_export.json     # optional export
    phase1_progression.json # optimize phase 1
    phase2_evolution.json   # optimize phase 2
```

## Testing

```powershell
conda run -n kernup-dev pytest
```

## Project Status

This repository currently ships a comprehensive dry-run architecture for iterative development and CI-friendly validation. Real GPU benchmarking and local LLM execution paths are scaffolded and intentionally marked as future implementation points.

## Contributing

See `CONTRIBUTING.md`.

## License

MIT (see `LICENSE`).
