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
- Phase 2 safety guard: prompt generation is blocked with `KernupError` if reference kernel is missing
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

### 1) Clone the repository

```bash
git clone https://github.com/keypaa/kernup.git
cd kernup
```

### 2) Create and activate an environment

Conda is recommended, especially on Windows.

Windows (PowerShell):

```powershell
conda create -y -n kernup-dev python=3.11
conda activate kernup-dev
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Linux/macOS (bash/zsh):

```bash
conda create -y -n kernup-dev python=3.11
conda activate kernup-dev
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

### 3) Verify installation

```bash
kernup --help
```

Expected command list includes: `profile`, `optimize`, `patch`, `bench`, `status`, `clean`.

### Optional: basic editable install without conda

```bash
python -m pip install -e .
```

## Quickstart

The current release is dry-run first. On CPU-only systems, always pass `--allow-no-gpu`.

### Quickstart A: full dry-run workflow (recommended)

1. Profile and create first run artifacts.

```powershell
kernup profile --hf Qwen/Qwen2.5-7B --dry-run --allow-no-gpu --output ./kernup_results --export
```

2. Run Phase 1 dry-run optimization.

```powershell
kernup optimize --hf Qwen/Qwen2.5-7B --phase 1 --dry-run --allow-no-gpu --iterations 10 --population 4
```

3. Run Phase 2 dry-run evolution.

```powershell
kernup optimize --hf Qwen/Qwen2.5-7B --phase 2 --dry-run --allow-no-gpu --iterations 6 --population 4
```

4. Generate a deployment patch from latest run.

```powershell
kernup patch --hf Qwen/Qwen2.5-7B --results ./kernup_results --format simple --output ./patch_out
```

`patch` validates that the selected run model matches `--hf`. Use `--allow-model-mismatch` only if you intentionally want to bypass this safeguard.

5. Generate benchmark-style summary export.

```powershell
kernup bench --hf Qwen/Qwen2.5-7B --results ./kernup_results --export --output ./kernup_results
```

`bench` applies the same model consistency check and supports `--allow-model-mismatch` for explicit override.

### Quickstart B: one-command smoke check script (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release_smoke_windows.ps1
```

This script runs tests plus `profile`, `optimize` (phase 1 and phase 2), `patch`, and `bench` in sequence.

## CLI Reference

### profile

```text
kernup profile --hf <repo/model> [--device cuda:0] [--dry-run] [--allow-no-gpu] [--export] [--output ./kernup_results]
```

### optimize

```text
kernup optimize --hf <repo/model> --phase 1|2 [--target throughput|latency|balanced] [--iterations N] [--population N] [--plateau-window N] [--plateau-threshold F] [--resume] [--dry-run] [--allow-no-gpu] [--allow-model-mismatch] [--output ./kernup_results]
```

When `--resume` is used, Kernup checks that the latest run model matches `--hf`. Use `--allow-model-mismatch` only if you intentionally want to bypass this guardrail.

### patch

```text
kernup patch --hf <repo/model> --results ./kernup_results --format simple|vllm|tgi|sglang --output ./patch [--allow-model-mismatch]
```

### bench

```text
kernup bench --hf <repo/model> --results ./kernup_results [--seq-lens 128,512,2048] [--batch-sizes 1,4,8,16] [--export] [--output ./kernup_results] [--allow-model-mismatch]
```

### utilities

```text
kernup status --results ./kernup_results [--hf <repo/model>]
kernup clean --results ./kernup_results [--yes]
```

`status` displays the run model id and can be filtered by model with `--hf`.

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
