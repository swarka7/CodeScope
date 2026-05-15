# CodeScope — AI-powered repository intelligence and failure-aware debugging foundation.

CodeScope is an experimental developer-tooling project that builds repository intelligence for Python codebases using AST analysis, semantic retrieval, dependency-aware context, and failure-aware debugging workflows.

## Status

CodeScope is currently under active development.
Core repository analysis, semantic retrieval, persistent indexing, and failure-aware diagnosis are implemented.
Autonomous repair, patch generation, safe diff application, and VS Code integration are planned future milestones.

## What is CodeScope?

CodeScope scans Python repositories, parses code with the Python AST, and extracts searchable code “chunks” (functions, classes, methods, imports, dependencies).
It uses embeddings for semantic retrieval and enriches results with simple dependency-aware context.
When tests fail, it converts failure output into a retrieval query and surfaces likely relevant code from the existing local index.

## Motivation

Large repositories are hard to navigate and even harder to debug under time pressure.
Semantic retrieval helps you search by intent, and lightweight static analysis provides structure and context around what you find.
CodeScope explores practical “repository intelligence” and failure-aware diagnosis workflows without assuming a fully autonomous repair system.

## Features

- Repository scanning
- AST-based code parsing
- Function/class/method chunk extraction
- Import/dependency/decorator extraction
- Semantic search with embeddings
- Multi-hop dependency-aware retrieval
- Persistent and incremental local indexing (`.codescope/`)
- Pytest runner
- Failure parsing
- Failure-aware diagnosis with rule-based summaries (`diagnose`)

## Architecture

```text
Repository
  ↓
Scanner
  ↓
AST Parser
  ↓
Chunker
  ↓
Embeddings
  ↓
Local Index
  ↓
Semantic + Dependency Retrieval
  ↓
Failure-Aware Diagnosis
```

## Current limitations

- Symbol resolution is heuristic/static (basic imports + aliases + same-file hints) and does not perform full Python type inference or dynamic import execution.
- Local indexes use readable JSON storage; incompatible index or embedding-text versions currently trigger rebuilds rather than migrations.

## Installation

Create a virtual environment and install CodeScope in editable mode:

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

python -m pip install -e ".[dev,ai]"
```

## CLI usage

Scan a repository:

```bash
python -m codescope.cli scan .
```

Extract chunks (structure view):

```bash
python -m codescope.cli chunks .
```

Build a persistent local index in `.codescope/`:

```bash
python -m codescope.cli index .
```

Search (requires an existing index):

```bash
python -m codescope.cli search . "repository scanner"
```

Run tests:

```bash
python -m codescope.cli test .
```

Diagnose failing tests and retrieve likely relevant code (requires an existing index when tests fail):

```bash
python -m codescope.cli diagnose .
```

## Demos

### Buggy calculator (minimal)

This repo includes a tiny intentionally broken project at `examples/buggy_calculator/`.

```bash
python -m codescope.cli index examples/buggy_calculator
python -m codescope.cli diagnose examples/buggy_calculator
```

Expected output (example):

```text
Tests failed
[FAIL] tests/test_calculator.py::test_calculate_discount_applies_percent
Message: calculate_discount(100, 10) returned -900 instead of 90

Diagnosis summary:
- Failing test: test_calculate_discount_applies_percent
- Failure signal: AssertionError, calculate_discount(100, 10) returned -900 instead of 90
- Most relevant source chunk: calculate_discount in calculator.py

Likely relevant code:
[semantic] [function] calculate_discount calculator.py:4-16
```

### Buggy auth service (more realistic)

For a more backend-style demo, see `examples/buggy_auth_service/` (token parsing + validation).

```bash
python -m codescope.cli index examples/buggy_auth_service
python -m codescope.cli diagnose examples/buggy_auth_service
```

The failing test calls `validate_token(...)` for an expired token and expects `False`, so diagnosis should ideally surface `validate_token` and related token parsing context (e.g. `decode_token`, `TokenPayload`).

## Testing

```bash
pytest
ruff check .
```

## Roadmap

- Source-aware ranking
- Improved traceback-to-source mapping
- Patch generation
- Safe diff application
- Test validation loop
- Iterative repair loop
- Qdrant/FAISS backend
- VS Code extension
