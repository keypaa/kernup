# Contributing to Kernup

Thank you for your interest in contributing.

## Development Setup

### 1. Clone and enter repo

```bash
git clone https://github.com/keypaa/kernup.git
cd kernup
```

### 2. Create Conda environment (recommended)

```powershell
conda create -y -n kernup-dev python=3.11
conda activate kernup-dev
python -m pip install -e .[dev]
```

### 3. Run tests

```powershell
pytest
```

## Contribution Workflow

1. Create a branch from `main`.
2. Keep changes focused and small.
3. Add or update tests for behavior changes.
4. Run test suite before opening a PR.
5. Open PR with a clear summary and validation notes.

## Coding Guidelines

- Python 3.11+
- Keep modules focused and testable.
- Prefer typed dataclasses and explicit interfaces.
- Keep dry-run support for CPU-only validation where possible.
- Avoid introducing hard dependencies on GPU-only execution in test paths.

## Commit Message Style

Use concise commit messages with a prefix:
- `feat:` new functionality
- `fix:` bug fix
- `docs:` documentation changes
- `test:` test-only updates
- `chore:` maintenance

## Reporting Issues

Include:
- OS and Python version
- Command executed
- Full error output
- Minimal reproduction steps

## Scope Notes

Current development emphasizes dry-run architecture and progressive module scaffolding. If you contribute a feature that needs GPU runtime, include a fallback, mock path, or skip logic for CPU-only CI.
