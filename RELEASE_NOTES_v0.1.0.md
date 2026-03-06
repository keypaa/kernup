# Kernup v0.1.0 Release Notes

Release date: 2026-03-06

## Summary

Kernup v0.1.0 delivers a complete dry-run-first implementation of the planned architecture, including profiling, phase 1 search, phase 2 validation/evolution scaffolding, patch generation, benchmarking summaries, storage, and contributor documentation.

This release is designed to be runnable and testable on CPU-only environments (including Windows) while preserving clean extension points for full GPU execution and local LLM integration.

## Highlights

- End-to-end CLI surface implemented:
  - `profile`
  - `optimize`
  - `patch`
  - `bench`
  - `status`
  - `clean`
- Run artifact persistence:
  - SQLite database (`kernup.db`)
  - logs (`kernup.log`)
  - structured JSON artifacts
- Phase 1 dry-run genetic search:
  - hyperparameter population + mutation
  - plateau detection
  - scoring cache scaffold
- Phase 2 dry-run pipeline:
  - static validation
  - numerical validation
  - benchmark gate (synthetic in dry-run)
  - healing retries
  - prompt budget builder
  - generator + evolution loop
- Patch output formats:
  - simple
  - vllm
  - tgi
  - sglang
- Bench summary command with export
- Test suite expanded and validated locally

## Validation Performed

- `conda run -n kernup-dev pytest` -> pass
- CLI smoke checks:
  - `profile --dry-run --allow-no-gpu`
  - `optimize --phase 1 --dry-run --allow-no-gpu`
  - `optimize --phase 2 --dry-run --allow-no-gpu`
  - `patch`
  - `bench --export`

## Known Limitations

- Real non-dry-run GPU benchmarking remains scaffolded, not finalized.
- Real local LLM generation path remains scaffolded, with dry-run deterministic generation active.
- Baseline speedup in `bench` is placeholder until real baseline persistence is fully integrated.

## Upgrade / Install

```powershell
conda create -y -n kernup-dev python=3.11
conda activate kernup-dev
python -m pip install -e .[dev]
```

## Recommended Next Work

- Implement real GPU execution paths for profile and optimize.
- Integrate real local HF code model generation in phase 2.
- Add CI workflow matrix and release automation.
