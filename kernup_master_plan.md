# KERNUP — Master Implementation Plan
> Automatic Kernel Optimization for HuggingFace Transformers  
> Version 0.1 — Hobby Project — MIT License

---

## Table of Contents
1. [Vision & Project Overview](#1-vision--project-overview)
2. [Code Architecture](#2-code-architecture)
3. [Initial Profiling](#3-initial-profiling)
4. [Phase 1 — Hyperparameter Search](#4-phase-1--hyperparameter-search)
5. [Phase 2 — Kernel Generation (Opt-in)](#5-phase-2--kernel-generation-opt-in)
6. [Storage & Results](#6-storage--results)
7. [CLI Reference](#7-cli-reference)
8. [User Configuration](#8-user-configuration)
9. [Error Handling & Logging](#9-error-handling--logging)
10. [Testing Strategy](#10-testing-strategy)
11. [Technical Stack](#11-technical-stack)
12. [Documentation & Repository](#12-documentation--repository)
13. [Implementation Order](#13-implementation-order)
14. [Future / Out of Scope for v0.1](#14-future--out-of-scope-for-v01)

---

## 1. Vision & Project Overview

Kernup is an open-source, pip-installable CLI tool that automatically optimizes HuggingFace transformer models for high-performance PyTorch inference. The user provides a model identifier and Kernup handles everything autonomously: profiling, hyperparameter search, kernel generation, and patch output.

### 1.1 Target Users

Kernup targets professionals doing high-throughput model serving with PyTorch-based frameworks (vLLM, TGI, SGLang). It is **NOT** designed for llama.cpp/GGUF users.

### 1.2 Core Philosophy

- Fully autonomous — one command, zero manual tuning required
- No quantization — pure BF16, no INT8/FP8
- No external API dependencies — 100% local LLM for kernel generation
- GPU-agnostic — works on any NVIDIA GPU (local or cloud)
- Modular and debuggable — every component is independently testable

### 1.3 Target Stack

| Component | Technology |
|-----------|------------|
| Inference framework | PyTorch + vLLM / TGI / SGLang |
| Kernel language | Triton (Python) |
| Model source | HuggingFace Hub |
| Hardware | NVIDIA GPU (any VRAM, local or cloud) |
| Precision | BF16 native only |
| LLM generator | Local small model (1-3B, e.g. Qwen2.5-Coder-1.5B) |

### 1.4 The Core Command

```bash
git clone https://github.com/<user>/kernup
cd kernup
pip install -e .
kernup optimize --hf Qwen/Qwen2.5-7B
```

That's all the user needs to type. Kernup handles the rest.

> **Note:** PyPI publishing is out of scope for v0.1. The goal is to build a clean, well-structured package first. PyPI comes later.

---

## 2. Code Architecture

### 2.1 Project Structure

```
kernup/
├── cli/
│   ├── __init__.py
│   ├── main.py          # entry point, argument parsing
│   ├── profile.py       # kernup profile command
│   ├── optimize.py      # kernup optimize command
│   ├── patch.py         # kernup patch command
│   ├── bench.py         # kernup bench command
│   └── status.py        # kernup status/clean commands
│
├── profiler/
│   ├── __init__.py
│   ├── hardware.py      # GPU detection (SM count, VRAM, bandwidth)
│   ├── baseline.py      # HuggingFace reference run
│   └── breakdown.py     # torch.profiler bottleneck analysis
│
├── phase1/
│   ├── __init__.py
│   ├── search.py        # genetic search engine
│   ├── hyperparams.py   # hyperparameter definitions and ranges
│   └── scoring.py       # tok/s, latency, VRAM evaluation
│
├── phase2/
│   ├── __init__.py
│   ├── evolution.py     # evolutionary engine (selection, mutation, elitism)
│   ├── generator.py     # local LLM interface
│   ├── prompt.py        # prompt construction (English only)
│   ├── healer.py        # self-healing logic (max 3 attempts)
│   └── validator/
│       ├── __init__.py
│       ├── static.py    # syntax validation
│       ├── numerical.py # numerical correctness validation
│       └── benchmark.py # full GPU benchmark
│
├── storage/
│   ├── __init__.py
│   ├── db.py            # SQLite interface
│   └── export.py        # optional JSON export
│
├── patch/
│   ├── __init__.py
│   ├── simple.py        # patch_model.py generator
│   ├── vllm.py          # vLLM native plugin
│   ├── tgi.py           # TGI native plugin
│   └── sglang.py        # SGLang native plugin
│
├── utils/
│   ├── __init__.py
│   ├── gpu.py           # CUDA/torch helpers
│   ├── model.py         # HuggingFace helpers (load, config parsing)
│   ├── display.py       # rich terminal output (tables, progress bars)
│   └── graph.py         # progression graph generation
│
└── config/
    ├── __init__.py
    ├── defaults.py      # all default values
    └── schema.py        # user config validation
```

### 2.2 Key Data Interfaces

All modules communicate via typed dataclasses:

```python
@dataclass
class GPUProfile:
    name: str
    sm_count: int
    sram_per_sm_kb: int
    hbm_bandwidth_tbs: float
    vram_total_gb: float
    vram_free_gb: float
    compute_capability: str

@dataclass
class BaselineResult:
    tok_s: float
    ttft_ms: float
    latency_ms: float
    vram_used_gb: float
    seq_len: int

@dataclass
class KernelScore:
    tok_s: float
    ttft_ms: float
    latency_ms: float
    vram_used_gb: float
    generation: int
    config: dict

@dataclass
class KernelVariant:
    code: str
    parent_id: str
    mutation_type: str  # light / medium / heavy
    generation: int
```

### 2.3 Full Run Flow

```
cli/optimize.py
    → profiler/hardware.py         # detect GPU
    → profiler/baseline.py         # measure baseline performance
    → profiler/breakdown.py        # identify bottlenecks
    → phase1/search.py             # hyperparameter tuning
        → phase1/hyperparams.py    # define ranges
        → phase1/scoring.py        # benchmark each config
        → storage/db.py            # save every result
    → [if --phase 2]
    → phase2/evolution.py          # evolutionary engine
        → phase2/generator.py      # LLM generates variants
            → phase2/prompt.py     # build prompt
        → phase2/validator/static.py     # syntax check
        → phase2/validator/numerical.py  # numerical check
        → phase2/validator/benchmark.py  # full benchmark
        → phase2/healer.py         # self-heal on failure
        → storage/db.py            # save results
    → utils/graph.py               # generate progression graph
    → storage/export.py            # JSON export if --export
    → cli/patch.py                 # generate patch if requested
```

---

## 3. Initial Profiling

### 3.1 Overview

Before any optimization begins, Kernup runs a full profiling pass. This is also available as a standalone command. The profiler determines the optimization order dynamically — no hardcoded priorities.

### 3.2 What is Measured

- **Hardware profile:** GPU name, SM count, SRAM/SM, HBM bandwidth, VRAM total/free
- **Baseline performance:** tok/s, TTFT, latency at seq lengths 128 / 512 / 2048
- **Operation breakdown** via `torch.profiler`: % time in attention, MLP, RMSNorm, embeddings, etc.
- **Architecture detection:** reads `config.json` from HuggingFace (GQA, MHA, MLA, MoE...)

### 3.3 Terminal Output Example

```
╔══════════════════════════════════════════╗
║         KERNUP — Initial Profiling       ║
╠══════════════════════════════════════════╣
║ GPU        : NVIDIA A100 80GB            ║
║ VRAM       : 79.2GB total / 72.1GB free  ║
╠══════════════════════════════════════════╣
║ Model      : Qwen/Qwen2.5-7B             ║
║ Baseline   : 47.3 tok/s / 312ms TTFT     ║
╠══════════════════════════════════════════╣
║ Bottlenecks detected:                    ║
║  1. Linear projections    → 71%          ║
║  2. Attention (GQA)       → 18%          ║
║  3. RMSNorm               → 6%           ║
║  4. Other                 → 5%           ║
╠══════════════════════════════════════════╣
║ Optimization order set automatically.   ║
║ Starting Phase 1...                      ║
╚══════════════════════════════════════════╝
```

### 3.4 Standalone Command

```bash
kernup profile --hf Qwen/Qwen2.5-7B --export
```

---

## 4. Phase 1 — Hyperparameter Search

### 4.1 Overview

Phase 1 tunes Triton autotune hyperparameters using a genetic search engine. No kernel code is generated or modified. This phase always runs before Phase 2 and provides the warm-start configuration for kernel generation.

### 4.2 Hyperparameters — Funnel Order

Optimization follows a funnel: biggest impact first. The exact order is set dynamically by the profiler bottleneck analysis.

| Priority | Parameter | Impact | Typical Range |
|----------|-----------|--------|---------------|
| 1 — Big knob | `BLOCK_SIZE` | Major | 16, 32, 64, 128, 256 |
| 2 — Big knob | `num_warps` | Major | 1, 2, 4, 8, 16 |
| 3 — Medium | `num_stages` | Moderate | 1, 2, 3, 4, 5 |
| 4 — Medium | KV cache strategy | Moderate | sliding, full, chunked |
| 5 — Medium | Tensor padding/alignment | Moderate | 64, 128, 256 |
| 6 — Fine | Split-K factor | Minor | 1, 2, 4, 8 |
| 7 — Fine | Prefetch distance | Minor | 1, 2, 3 |
| 8 — Fine | L2 cache hints | Minor | evict_first, evict_last, none |

### 4.3 Kernel Compilation Caching

Triton recompiles kernels on every run by default. Over 100 iterations this is catastrophically slow. `phase1/scoring.py` must implement a hash-based cache:

- Compute a hash of `(kernel_code + hyperparameters + gpu_compute_capability)`
- If the hash matches a previous run, load the cached `.so` (shared object) instead of recompiling
- Cache stored per run in `kernup_results/run_xxx/.triton_cache/`

This turns repeated compilation (seconds each) into a cache lookup (milliseconds).

### 4.4 Genetic Search Engine

- **Tournament selection:** K=3 candidates, best wins. Maintains diversity.
- **Elitism:** best absolute config from all generations is always preserved.
- **Population:** 4–8 configs per generation (configurable, default 4).
- **Max iterations:** configurable (default 100).

### 4.5 Plateau Detection

Automatic stop when improvement stagnates:

```python
if (best_score_gen_N - best_score_gen_N-10) / best_score_gen_N-10 < 0.01:
    # plateau detected → stop
```

Window size (default: 10) and threshold (default: 1%) are both configurable. If max iterations is reached before plateau, a graph is shown so the user can decide whether to `--resume`.

### 4.6 Phase 1 Output

- Summary stats: before/after tok/s, TTFT, latency, VRAM
- Progression graph (generation vs best tok/s)
- Best config saved to SQLite and optionally exported to JSON

---

## 5. Phase 2 — Kernel Generation (Opt-in)

### 5.1 Overview

Phase 2 uses a local LLM to generate, mutate, and evolve Triton kernels. It is opt-in — the user sees Phase 1 results first and decides whether to go further. Phase 2 always warm-starts from the best Phase 1 configuration.

### 5.2 VRAM Strategy

- **Primary:** small LLM (1-3B, e.g. `Qwen2.5-Coder-1.5B`) runs alongside the benchmark model — parallel execution on T4 (16GB).
- **Reserve option:** 7B LLM + time-slicing (LLM generates → unload → load model → benchmark → repeat). Slower but higher quality generation.

**Automatic VRAM guard:** Before starting Phase 2, Kernup computes:

```python
if (model_vram_gb + llm_vram_gb + estimated_overhead_gb) > gpu_vram_free_gb:
    # force timeslice regardless of config, warn the user
    llm_load_strategy = "timeslice"
```

The estimated overhead (KV cache, activations, OS) defaults to 3GB. This prevents OOM crashes silently — the user sees a clear warning explaining why timeslice was forced.

### 5.3 Validation Pipeline

Every generated kernel passes through 3 sequential gates before benchmarking:

| Stage | What is checked | GPU Cost | On failure |
|-------|----------------|----------|------------|
| 1 — Static | Python syntax, Triton signatures, dimension consistency | Zero | Self-heal → new kernel |
| 2 — Numerical | Output matches HuggingFace reference within epsilon (micro-input: 16 tokens) — uses `torch.testing.assert_close` with relaxed BF16 tolerances (`rtol=1e-2, atol=1e-2`) | Low | Self-heal → new kernel |
| 3 — Benchmark | Full tok/s, latency, VRAM on realistic sequences | High | Discard, score = 0 |

**Benchmark warm-up (Stage 3):** Before measuring, always run 3 warm-up passes to let Triton compile and the GPU reach steady state. Only then start timing. This avoids measuring kernel launch overhead instead of actual throughput.

**Self-healing:** on failure at stage 1 or 2, the error message is sent back to the LLM for correction (max 3 attempts). After 3 failed attempts, a completely new kernel is requested.

### 5.4 Evolutionary Engine

- **Tournament selection:** K=3, same as Phase 1.
- **Elitism:** best kernel of all generations is always preserved.
- **Population:** 4–8 kernels per generation (default 4).
- **Mutation types:**
  - Light (60% default): change one specific aspect — *"modify only the tiling strategy"*
  - Medium (30% default): rewrite one block — *"rewrite only the memory access pattern"*
  - Heavy (10% default): full rewrite using parent as inspiration
- **Adaptive mutation:** if plateau detected for 5+ generations, heavy mutation ratio increases automatically to escape local optimum.
- **Plateau detection:** same mechanism as Phase 1 (configurable window + threshold).

### 5.5 LLM Context Budget

All prompts are written in **English**. Context is strictly budgeted for small models (4k-8k window):

| Block | Content | Approx tokens |
|-------|---------|---------------|
| 1 | Role, constraints, rules (no quantization, BF16 only, code only) | ~100 |
| 2 | GPU hardware profile (name, SM count, SRAM, HBM bandwidth) | ~50 |
| 3 | Model architecture (arch, hidden size, heads, attention type, intermediate size) | ~100 |
| 4 | Best Phase 1 hyperparameters + score | ~100 |
| 5 | Reference kernel to improve | ~500-800 |
| 6 | Best kernel from previous generations + score (only 1) | ~500-800 |
| 7 | Final instruction (target metric, GPU-specific autotune configs) | ~100 |
| **TOTAL** | | **~1500-2000** |

**Mandatory guard (`phase2/prompt.py`):** if the reference kernel block is empty or missing, prompt generation must be blocked and a clear `KernupError` must be raised. Kernup must never send an incomplete prompt to the local LLM.

### 5.6 Self-Healing Prompt Addition

When self-healing is triggered, Block 7 is replaced with:

```
Your previous kernel failed at stage {stage} with this error:
{error_message}
{numerical_diff if stage 2}
Fix only this issue. Keep everything else identical.
```

### 5.7 Optimization Targets

- `--target throughput` — maximize tok/s, prioritize memory efficiency and batching
- `--target latency` — minimize TTFT, prioritize reducing kernel launch count
- `--target balanced` — optimize tok/s / latency ratio **(default)**

---

## 6. Storage & Results

### 6.1 SQLite Schema

All results are stored in a single SQLite file per run. Two tables linked by `run_id`:

```sql
CREATE TABLE runs (
    id          TEXT PRIMARY KEY,
    timestamp   TEXT,
    generation  INTEGER,
    block_size  INTEGER,
    num_warps   INTEGER,
    num_stages  INTEGER,
    kv_strategy TEXT,
    split_k     INTEGER,
    mutation_type TEXT
);

CREATE TABLE results (
    id           TEXT PRIMARY KEY,
    run_id       TEXT,
    tok_s        REAL,
    ttft_ms      REAL,
    latency_ms   REAL,
    vram_used_gb REAL,
    is_best      INTEGER,
    notes        TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
```

### 6.2 Run Versioning

Every `kernup optimize` invocation creates a new run. Existing results are **NEVER** overwritten. Run IDs use a timestamp + short random suffix:

```
run_20250305_143022_a3f9b
```

### 6.3 Output Folder Structure

```
kernup_results/
└── run_20250305_143022_a3f9b/
    ├── kernup.log        # full run log
    ├── kernup.db         # SQLite results
    ├── profile.json      # GPU profile + baseline
    └── graph.png         # progression curve
```

### 6.4 JSON Export

Optional at the end of any run via `--export` flag. Provides a human-readable snapshot of all results.

---

## 7. CLI Reference

### 7.1 `kernup profile`

```bash
kernup profile --hf <repo/model>
               --device cuda:0
               --seq-lens 128,512,2048
               --export
               --output ./kernup_results
               --dry-run
```

### 7.2 `kernup optimize`

```bash
kernup optimize --hf <repo/model>
                --phase 1|2                          # default: both
                --target throughput|latency|balanced # default: balanced
                --iterations 100
                --population 4
                --plateau-window 10
                --plateau-threshold 0.01
                --resume
                --export
                --output ./kernup_results
                --device cuda:0
                --llm <repo/model>                   # default: Qwen/Qwen2.5-Coder-1.5B
                --mutation-ratio 60,30,10
                --dry-run
```

### 7.3 `kernup patch`

```bash
kernup patch --hf <repo/model>
             --results ./kernup_results
             --format simple|vllm|tgi|sglang         # default: simple
             --output ./patch
```

### 7.4 `kernup bench`

```bash
kernup bench --hf <repo/model>
             --results ./kernup_results
             --seq-lens 128,512,2048
             --batch-sizes 1,4,8,16
             --export
             --output ./kernup_results
```

### 7.5 Utility Commands

```bash
kernup status  --results ./kernup_results
kernup clean   --results ./kernup_results
kernup --version
kernup --help
kernup <command> --help
```

### 7.6 `--dry-run`

Available on `profile` and `optimize`. Validates config, detects GPU, verifies HuggingFace model accessibility — without touching the GPU. Essential for checking setup before spending cloud credits.

---

## 8. User Configuration

### 8.1 Config File Location

Global config lives at `~/.kernup/config.toml`. Created automatically on first run. CLI flags always override config file values.

### 8.2 `config.toml` Structure

```toml
[user]
timezone = "Europe/Paris"  # overrides machine local time if set

[defaults]
device = "cuda:0"
output = "./kernup_results"
target = "balanced"
iterations = 100
population = 4

[phase1]
plateau_window = 10
plateau_threshold = 0.01

[phase2]
llm = "Qwen/Qwen2.5-Coder-1.5B"  # any HuggingFace code model
llm_max_vram_gb = 4               # VRAM budget allocated to the generator
llm_load_strategy = "parallel"    # parallel = small model runs alongside benchmark
                                  # timeslice = unload/reload between steps (big models)
mutation_ratio = [60, 30, 10]
max_healing_attempts = 3
adaptive_plateau_window = 5

[export]
auto_export = false
format = "json"
```
### 8.3 Timezone

Default: machine local time via `datetime.now().astimezone()`. If `timezone` is set in `config.toml`, it overrides the machine setting. All run timestamps use this timezone.

### 8.4 LLM Generator Runner

The kernel generator LLM runs via **HuggingFace `transformers`** directly — already a core dependency of Kernup, zero extra installs, and compatible with any model on the HuggingFace Hub.

| `llm_load_strategy` | When to use | How it works |
|---------------------|-------------|--------------|
| `parallel` | Small models (1-3B), big GPUs | LLM stays loaded alongside the benchmark model |
| `timeslice` | Larger models (7B+), tighter VRAM | LLM loads → generates → unloads → benchmark runs → repeat |

**Examples by hardware:**

- T4 (16GB) → `llm = "Qwen/Qwen2.5-Coder-1.5B"`, `llm_load_strategy = "parallel"`
- A100 (80GB) → `llm = "Qwen/Qwen2.5-Coder-7B"`, `llm_load_strategy = "parallel"`
- H100 (80GB) → `llm = "Qwen/Qwen2.5-Coder-14B"`, `llm_load_strategy = "timeslice"`

---

## 9. Error Handling & Logging

### 9.1 Error Types

| Type | Cause | User sees | Stack trace in log |
|------|-------|-----------|-------------------|
| `UserError` | Wrong model name, insufficient VRAM, no NVIDIA GPU, invalid flags | Clear human message + suggestion | No |
| `KernupError` | Internal bug, unexpected crash, validation logic failure | Clear message + GitHub issue invite | Yes (in `kernup.log`) |

### 9.2 Log Levels

- `DEBUG` — detailed internal state, kernel code, raw scores (visible with `--verbose`)
- `INFO` — normal run progress, generation summaries (default)
- `WARNING` — non-fatal issues, fallback behaviors
- `ERROR` — failures, validation errors, crashes

### 9.3 Log File

Every run writes a `kernup.log` inside its result folder. Log level in file is always `DEBUG` regardless of terminal verbosity — useful for post-run debugging.

### 9.4 Common UserErrors to Handle Explicitly

- Model not found on HuggingFace Hub → suggest checking the model identifier
- Insufficient VRAM to load model → show required vs available VRAM
- No NVIDIA GPU detected → remind that NVIDIA GPU is required
- Triton not supported on detected GPU → show compute capability requirement
- Phase 2 requested without Phase 1 results → suggest running Phase 1 first

---

## 10. Testing Strategy

### 10.1 Unit Tests

- `profiler/hardware.py` — mock `torch.cuda`, test `GPUProfile` dataclass population
- `phase1/hyperparams.py` — test ranges, boundary values, invalid configs
- `phase2/validator/static.py` — test valid/invalid Triton syntax cases
- `phase2/validator/numerical.py` — test epsilon comparison logic
- `phase2/healer.py` — test retry logic, max attempts, fallback to new kernel
- `phase2/prompt.py` — test context budget enforcement, English-only output
- `storage/db.py` — test read/write/query on in-memory SQLite
- `utils/gpu.py` — test helpers with mocked CUDA

### 10.2 Integration Tests

- Full Phase 1 run on a tiny test model (GPT-2 or similar) — verifies end-to-end flow
- Full Phase 2 run with mocked LLM generator — verifies validation pipeline
- `--resume` test — verifies checkpoint loading and continuation
- `--dry-run` test — verifies no GPU is touched
- Export test — verifies JSON output matches SQLite contents

### 10.3 Test Infrastructure

- `pytest` as test runner
- `unittest.mock` for GPU and HuggingFace mocking
- Fixtures for `GPUProfile`, `KernelScore`, `BaselineResult` dataclasses
- CI-friendly: unit tests must run without a GPU

---

## 11. Technical Stack

| Package | Purpose | Notes |
|---------|---------|-------|
| `click` | CLI framework | More ergonomic than argparse |
| `rich` | Terminal UI (tables, progress bars, colors) | |
| `torch` | PyTorch + CUDA | Core requirement |
| `triton` | Triton kernels | Pin `>=3.1,<4.0` — breaks frequently across versions |
| `transformers` | HuggingFace model loading | >=4.40 |
| `sqlite3` | Results storage | Built-in Python, zero extra dep |
| `matplotlib` | Progression graphs | |
| `optuna` | Optional Bayesian optimization for Phase 1 | Can replace genetic search |
| `dataclasses` | Typed interfaces between modules | Built-in Python |
| `pytest` | Test runner | |
| `tomllib` | Config file parsing | Built-in Python 3.11+ |

---

## 12. Documentation & Repository

### 12.1 Files to Create

- `README.md` — project overview, installation, quickstart, CLI reference, examples
- `CONTRIBUTING.md` — how to contribute, code style, PR process, test requirements
- `CHANGELOG.md` — version history (start at 0.1.0)
- `LICENSE` — MIT
- `pyproject.toml` — packaging config (ready for future PyPI publish)

### 12.2 README Structure

1. What is Kernup? (one paragraph)
2. Requirements (NVIDIA GPU, Python 3.11+, PyTorch 2.6+)
3. Installation
4. Quickstart (the two commands)
5. CLI reference
6. Example output
7. Architecture overview
8. Contributing
9. License

### 12.3 License

MIT License — same as RightNow AI and most HuggingFace ecosystem tools.

---

## 13. Implementation Order

Build in this order to always have a working, testable state at each step:

| Step | What to build | Why first |
|------|--------------|-----------|
| 1 | Project skeleton: `pyproject.toml`, `config/`, `utils/model.py`, `utils/display.py` | Foundation for everything |
| 2 | Storage: `storage/db.py` + `storage/export.py` | Needed by all other modules |
| 3 | Profiler: `hardware.py` + `baseline.py` + `breakdown.py` | Drives optimization order |
| 4 | CLI: `kernup profile` command end-to-end | First usable command |
| 5 | Phase 1: `hyperparams.py` + `scoring.py` + `search.py` | Core value of the tool |
| 6 | CLI: `kernup optimize --phase 1` end-to-end | Second usable command |
| 7 | Graphs: `utils/graph.py` + plateau detection | Makes results interpretable |
| 8 | Phase 2: `validator/` + `healer.py` + `prompt.py` | Validation before generation |
| 9 | Phase 2: `generator.py` + `evolution.py` | Full Phase 2 engine |
| 10 | CLI: `kernup optimize --phase 2` end-to-end | Full optimize command |
| 11 | Patch: `simple.py` + `vllm.py` + `tgi.py` + `sglang.py` | Output delivery |
| 12 | CLI: `kernup patch` + `kernup bench` + utilities | Complete CLI |
| 13 | Tests: unit + integration | Validate everything |
| 14 | Documentation: README + CONTRIBUTING + CHANGELOG | Ready to share |

---

## 14. Future / Out of Scope for v0.1

- Multi-GPU support (`--device cuda:0,cuda:1`)
- Multi-island evolutionary engine (MAP-Elites with 4+ islands)
- 7B LLM generator with time-slicing option
- PyPI publish (`pip install kernup`)
- llama.cpp / GGUF parameter optimization scripts (separate optional scripts)
- AMD GPU support
- Web dashboard for results visualization
