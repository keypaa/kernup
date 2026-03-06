# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-03-06

### Added
- Initial installable package structure with `kernup` CLI entrypoint.
- Error taxonomy with `UserError` and `KernupError`.
- Profiling dry-run workflow with run artifact generation.
- SQLite storage layer and JSON export helpers.
- Shared typed dataclasses for core interfaces.
- Phase 1 dry-run hyperparameter search, scoring cache scaffold, and optimize flow.
- Phase 2 validation-first modules (static, numerical, benchmark), healer, and prompt budget builder.
- Phase 2 dry-run generator and evolution loop integration.
- Patch command with `simple`, `vllm`, `tgi`, `sglang` formats.
- Bench command for summary display and JSON export.
- Utility commands: status and clean.
- Expanded test suite for CLI, storage, phase1, phase2 pipeline, and evolution.

### Notes
- Non-dry-run GPU execution and real local LLM generation are scaffolded and intentionally not finalized in this release.
