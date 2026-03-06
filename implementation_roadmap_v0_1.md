# KERNUP v0.1 Implementation Roadmap

This roadmap turns the master plan into an execution sequence with clear outputs and done criteria.

## Guiding Principles
- Keep the repository runnable at every step.
- Prefer small vertical slices over large unfinished modules.
- Build test scaffolding early so core logic remains safe to refactor.
- Keep Phase 2 optional until Phase 1 is stable and benchmarked.

## Milestone 0 - Bootstrap (Day 0-1)
Goal: repository is installable and CLI entry point works.

Deliverables:
- Python package skeleton matching planned module layout.
- pyproject.toml with dependencies and console script for kernup.
- Base error classes (UserError, KernupError).
- Minimal CLI root command with help output.
- Default config loader and schema validator stubs.

Done criteria:
- pip install -e . succeeds.
- kernup --help runs without error.
- pytest runs (even with placeholder tests).

## Milestone 1 - Storage Core (Day 1-2)
Goal: every run can be persisted and queried.

Deliverables:
- storage/db.py with SQLite schema creation and CRUD helpers.
- Run folder creation helper with run id format.
- storage/export.py for JSON export from DB records.
- Dataclasses module for shared typed interfaces.

Done criteria:
- A test inserts and reads run + result rows in in-memory SQLite.
- Run folder contains kernup.db and kernup.log for a dry run.

## Milestone 2 - Profiling Vertical Slice (Day 2-4)
Goal: complete kernup profile command end to end.

Deliverables:
- profiler/hardware.py GPU detection and capability capture.
- profiler/baseline.py baseline throughput and latency measurement.
- profiler/breakdown.py torch.profiler bottleneck summary.
- cli/profile.py wired to output rich table summary and optional export.
- --dry-run path that validates setup without touching benchmark flow.

Done criteria:
- kernup profile --hf <model> --dry-run passes locally.
- Profiling command produces profile.json and logs under run folder.
- User-facing errors are clear for missing model/GPU/VRAM.

## Milestone 3 - Phase 1 Search Engine (Day 4-7)
Goal: optimize hyperparameters with measurable improvement tracking.

Deliverables:
- phase1/hyperparams.py parameter space and validation.
- phase1/scoring.py benchmark scoring with triton compile cache hash.
- phase1/search.py genetic loop (tournament, elitism, plateau stop).
- utils/graph.py generation curve rendering.
- cli/optimize.py with phase 1 path.

Done criteria:
- kernup optimize --phase 1 runs from profile to best config save.
- Plateau detection stops early when improvement threshold not met.
- graph.png is generated and best config marked in DB.

## Milestone 4 - Optimize Orchestration Hardening (Day 7-8)
Goal: make phase 1 flow robust and resumable.

Deliverables:
- Resume support from latest incomplete run.
- Consistent structured logging across modules.
- Status command to inspect runs and current best result.
- Clean command for removing old run artifacts.

Done criteria:
- Interrupted phase 1 run can continue with --resume.
- status and clean commands work against output directory.

## Milestone 5 - Phase 2 Validation First (Day 8-10)
Goal: build safe validation pipeline before generation loop.

Deliverables:
- phase2/validator/static.py syntax and signature checks.
- phase2/validator/numerical.py BF16 tolerance checks.
- phase2/validator/benchmark.py warmup + benchmark protocol.
- phase2/healer.py retry state machine and max-attempt fallback.
- phase2/prompt.py context budgeting and English-only enforcement.

Done criteria:
- Validator tests cover pass/fail behavior at each stage.
- Healer stops after configured attempts and triggers new variant request.

## Milestone 6 - Phase 2 Generation Loop (Day 10-13)
Goal: opt-in kernel evolution works with local HF model.

Deliverables:
- phase2/generator.py transformers-based local generation.
- phase2/evolution.py population loop and adaptive mutation ratio.
- VRAM guard logic for parallel vs timeslice strategy.
- cli/optimize.py phase 2 integration and target mode support.

Done criteria:
- kernup optimize --phase 2 executes complete loop with validation gates.
- Forced timeslice warning appears when VRAM guard triggers.

## Milestone 7 - Patch Outputs + Bench Command (Day 13-15)
Goal: deliver usable integration artifacts for serving stacks.

Deliverables:
- patch/simple.py patch model script generation.
- patch/vllm.py, patch/tgi.py, patch/sglang.py plugin outputs.
- cli/patch.py and cli/bench.py commands.

Done criteria:
- Patch files are generated from best stored result.
- Bench command compares baseline vs optimized results by seq and batch.

## Milestone 8 - Test and Docs Completion (Day 15-17)
Goal: repository is shareable and contributor-ready.

Deliverables:
- Unit tests for all core modules listed in master plan.
- Integration tests for profile, phase 1, dry run, export, resume.
- README, CONTRIBUTING, CHANGELOG, LICENSE.

Done criteria:
- Unit tests pass on CPU-only CI.
- Documentation supports first-time setup and quickstart.

## Cross-Cutting Implementation Notes
- Keep dataclasses in one central module to avoid circular imports.
- Add protocol interfaces for benchmark and generator components to ease mocking.
- Store all run artifacts under one run root to simplify cleanup and reproducibility.
- Separate user-facing exception messages from internal stack traces.

## Recommended Initial File Creation Batch
Create these first to unlock vertical progress:
- kernup/__init__.py
- kernup/cli/main.py
- kernup/config/defaults.py
- kernup/config/schema.py
- kernup/storage/db.py
- kernup/storage/export.py
- kernup/types.py
- tests/test_smoke_cli.py

## First 3 Execution Tasks
1. Scaffold package + pyproject + CLI root command.
2. Implement storage layer and run artifact directory creation.
3. Implement kernup profile --dry-run from CLI to structured output.

## Definition of Ready for Coding
- Python version pinned (3.11+).
- Dependency ranges agreed (torch/triton/transformers).
- Output directory contract finalized (run id, filenames).
- Error taxonomy agreed (UserError vs KernupError).

## Definition of Done for v0.1
- profile, optimize, patch, bench, status, clean commands are functional.
- Phase 1 is stable and can improve baseline on at least one test model.
- Phase 2 is opt-in and guarded by validation and VRAM checks.
- Tests and docs are sufficient for external contributors to run project locally.
