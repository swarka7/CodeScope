# CodeScope — AI-powered repository intelligence and failure-aware debugging foundation.

CodeScope is an experimental developer-tooling project that builds repository intelligence for Python codebases using AST analysis, semantic retrieval, dependency-aware context, and failure-aware debugging workflows.

## Status

CodeScope is currently under active development.
Core repository analysis, semantic retrieval, persistent indexing, and failure-aware diagnosis are implemented.
Autonomous repair, patch generation, safe diff application, and VS Code integration are planned future milestones.

## What is CodeScope?

CodeScope scans Python repositories, parses code with the Python AST, and extracts searchable code “chunks” (functions, classes, methods, imports, dependencies).
It builds a persistent semantic index, enriches matches with dependency-aware context, and keeps search grounded in source structure.
When pytest fails, it turns failure output into a retrieval query, reranks likely source chunks, and prints deterministic reasons for why each result was selected.

## Motivation

Large repositories are hard to navigate and even harder to debug under time pressure.
Semantic retrieval helps you search by intent, and lightweight static analysis provides structure and context around what you find.
CodeScope explores practical “repository intelligence” and failure-aware diagnosis workflows without assuming a fully autonomous repair system.

## Project philosophy

- Retrieval quality first
- Explainability first
- Deterministic behavior first
- Generalized heuristics before LLM reasoning
- Patch generation much later, after trustworthy context retrieval

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
- Deterministic retrieval reasons for diagnose results

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

## How it works

- **Repository scanning** discovers Python source files while ignoring generated/cache directories.
- **AST chunk extraction** turns modules into function, class, and method chunks with imports, decorators, and dependencies.
- **Semantic indexing** embeds chunk text and stores readable local JSON under `.codescope/`.
- **Dependency-aware retrieval** expands semantic matches with directly and indirectly related chunks.
- **Failure-aware pytest diagnosis** parses failing tests, builds a focused query, and reranks likely source code.
- **Deterministic retrieval reasons** explain each result using rule-based signals such as expected exceptions, validation helpers, traceback files, and keyword overlap.

CodeScope currently does **not** generate patches or apply fixes. The current focus is trustworthy retrieval and debugging context before automated repair.

## Try it locally

Run the auth service demo to see indexing, semantic search, and failure-aware diagnosis:

```bash
python -m codescope.cli index examples/buggy_auth_service
python -m codescope.cli search examples/buggy_auth_service "expired token validation"
python -m codescope.cli diagnose examples/buggy_auth_service
```

The diagnosis output should include `CodeScope Diagnose`, `Diagnosis summary`, `Possible issue`, `Likely relevant code`, `Related context`, and deterministic `reasons=...` explanations.
See the full walkthrough in [`docs/demo.md`](docs/demo.md).

## Current limitations

- Symbol resolution is heuristic/static (basic imports + aliases + same-file hints) and does not perform full Python type inference or dynamic import execution.
- Local indexes use readable JSON storage; incompatible index or embedding-text versions currently trigger rebuilds rather than migrations.
- Failure diagnosis is rule-based and retrieval-focused; it can suggest likely relevant code, but it does not prove root cause or modify files.

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

More detail: see [`docs/failure_aware_diagnosis.md`](docs/failure_aware_diagnosis.md).

## Failure-aware diagnosis output

Diagnose output is designed to show the failure, a concise summary, a cautious rule-based hypothesis, and the code chunks CodeScope retrieved with deterministic reasons:

```text
CodeScope Diagnose

Status
- Tests failed

Failing test
- [FAIL] tests/test_auth_service.py::test_expired_token_is_rejected
- File: tests/test_auth_service.py:12

Failure signal
- Error: AssertionError
- Message: assert True is False

Diagnosis summary:
- Failing test: test_expired_token_is_rejected
- Failure signal: AssertionError, assert True is False
- Most relevant source chunk: validate_token in auth_service.py
- Related context: decode_token, TokenPayload
- Why: failure/query symbols overlap with retrieved source/context chunks.

Possible issue:
- validate_token may contain boolean validation logic returning the opposite truth value from what the test expects.
- This is a hypothesis based on the failure signal and retrieved code, not a proven root cause.

Likely relevant code:
1. validate_token
   Kind: function
   Location: auth_service.py:12-20
   Source: semantic
   Score: 1.42
   reasons=
     - validation logic
     - traceback file match
     - keyword match: expired, rejected

Related context:
1. decode_token
   Kind: function
   Location: token_manager.py:8-15
   Source: related
   reasons=
     - semantic match
```

The exact scores and ordering may change as the retrieval heuristics improve, but the output contract is intentionally explainable and deterministic.

## Demos

### Buggy calculator (minimal)

This repo includes a tiny intentionally broken project at `examples/buggy_calculator/`.

```bash
python -m codescope.cli index examples/buggy_calculator
python -m codescope.cli diagnose examples/buggy_calculator
```

Expected output (example):

```text
CodeScope Diagnose

Status
- Tests failed

Failing test
- [FAIL] tests/test_calculator.py::test_calculate_discount_applies_percent

Failure signal
- Message: calculate_discount(100, 10) returned -900 instead of 90

Diagnosis summary:
- Failing test: test_calculate_discount_applies_percent
- Failure signal: AssertionError, calculate_discount(100, 10) returned -900 instead of 90
- Most relevant source chunk: calculate_discount in calculator.py

Likely relevant code:
1. calculate_discount
   Kind: function
   Location: calculator.py:4-16
   Source: semantic
```

### Buggy auth service (more realistic)

For a more backend-style demo, see `examples/buggy_auth_service/` (token parsing + validation).

```bash
python -m codescope.cli index examples/buggy_auth_service
python -m codescope.cli diagnose examples/buggy_auth_service
```

The failing test calls `validate_token(...)` for an expired token and expects `False`, so diagnosis should ideally surface `validate_token` and related token parsing context (e.g. `decode_token`, `TokenPayload`).

### Buggy task API (multi-file backend validation)

For a more realistic multi-file backend example, see `examples/buggy_task_api/`.
It includes a task model, in-memory repository, service layer, validation helpers, route-like handlers, ownership checks, and task status transitions.
The intentional bug allows an archived task to be marked done, even though archived tasks should be terminal.

```bash
python -m codescope.cli index examples/buggy_task_api
python -m codescope.cli diagnose examples/buggy_task_api
```

This demo is useful for validating business-rule failures where an invalid operation is accepted instead of rejected.

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
