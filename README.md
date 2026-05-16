# CodeScope — Find the code that caused your failing test

A retrieval-first Python debugging tool that indexes a repository, runs pytest, and ranks likely root-cause code with deterministic explanations.

When a test fails, CodeScope helps answer one practical question:

> Which code should I inspect first?

CodeScope is not an LLM wrapper and does not automatically fix bugs. It builds the retrieval and debugging-context layer first: source-aware code search, dependency/call-path expansion, failure-aware ranking, and concise reasons for every result.

## Why This Matters

- Failing tests usually show symptoms, not root causes.
- Large repositories make it hard to know where to start.
- LLMs are useful, but they need focused, trustworthy context.
- CodeScope retrieves and explains that context before any optional LLM reasoning.

## What CodeScope Does

- Scans Python repositories.
- Extracts AST-based chunks for functions, classes, and methods.
- Builds a persistent semantic index.
- Runs pytest and parses failures.
- Diagnoses failures against the existing code index.
- Follows dependency, call-path, and reverse-call context.
- Ranks likely relevant source code.
- Explains why each result matters using deterministic reasons.

## Quick Demo

Example from the realistic banking benchmark:

```text
Failing test:
tests/test_transfers.py::test_successful_transfer_moves_money_and_records_activity

Likely relevant code:
1. TransferService.transfer
   Location: app/service.py:22-45
   Source: semantic
   reasons=
     - operation match: balance, record, transfer
     - business operation
     - state update logic
     - paired state operation

2. Account.credit
   Location: app/models.py:19-20
   Source: semantic
   reasons=
     - paired state operation
     - possible missing counterpart operation
```

The actual bug: the transfer flow debits the sender but does not credit the receiver. CodeScope ranks the business method first and surfaces the missing counterpart operation as useful context.

## v0.1 Benchmark Status

PASS means the expected root-cause chunk appears in the top 3 likely relevant code results.
These are small realistic benchmarks, not proof of production readiness.

| benchmark | bug | expected root cause | observed rank | result |
| --- | --- | --- | --- | --- |
| `banking_app` | transfer debits sender but does not credit receiver | `TransferService.transfer` | rank 1 | PASS |
| `movie_platform` | combined search filters ignore genre | `MovieSearchService.search` | rank 1 | PASS |
| `inventory_app` | insufficient-stock shipment is allowed | `FulfillmentService.ship_order` | rank 3 | PASS |

Full benchmark report: [`docs/benchmark_results.md`](docs/benchmark_results.md).

## Technical Highlights

- **AST chunking** extracts source structure without executing user code.
- **Semantic embeddings** support intent-based code search.
- **Persistent local indexing** stores readable JSON under `.codescope/`.
- **Incremental indexing** reuses unchanged file embeddings.
- **Atomic index writes** reduce partial/corrupt index risk.
- **Index compatibility/versioning** rejects stale indexes after schema or embedding-text changes.
- **Dependency-aware retrieval** expands semantic matches with nearby code relationships.
- **Static symbol resolution** handles common imports, aliases, relative imports, and same-class methods.
- **Call-path and reverse-call context** surfaces validators, callers, guards, and related business logic.
- **ScoreBreakdown-backed explanations** align ranking evidence with user-facing reasons.
- **Source-first ranking** keeps failing tests visible while prioritizing source code.
- **Business-behavior ranking** favors services, validators, search/filter logic, and state mutations over plumbing.
- **Paired state-operation reasoning** helps identify missing counterpart operations such as debit/credit-style bugs.
- **Deterministic output** keeps ranking and explanations auditable and testable.

## Usage

Install in editable mode:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

python -m pip install -e ".[dev,ai]"
```

Index a repository:

```bash
python -m codescope.cli index <repo>
```

Search indexed code:

```bash
python -m codescope.cli search <repo> "status transition validation"
```

Run tests:

```bash
python -m codescope.cli test <repo>
```

Diagnose failing tests:

```bash
python -m codescope.cli diagnose <repo>
```

Try a realistic benchmark:

```bash
python -m codescope.cli index examples/realistic_bugs/banking_app
python -m codescope.cli diagnose examples/realistic_bugs/banking_app
```

More details:

- Demo walkthrough: [`docs/demo.md`](docs/demo.md)
- Failure-aware diagnosis: [`docs/failure_aware_diagnosis.md`](docs/failure_aware_diagnosis.md)
- Benchmark results: [`docs/benchmark_results.md`](docs/benchmark_results.md)

## Architecture

```text
scan repo
  ↓
parse AST
  ↓
extract chunks
  ↓
build embedding text
  ↓
index
  ↓
semantic + dependency retrieval
  ↓
run pytest
  ↓
failure-aware ranking
  ↓
diagnosis summary + retrieval reasons
```

## Philosophy

- Retrieval quality first.
- Explainability first.
- Deterministic behavior first.
- Benchmarks before claims.
- LLM reasoning later, on top of retrieved context.
- Patch generation later, not now.

## Limitations

- Python-focused.
- Static analysis only; no runtime tracing.
- No automatic fixes or patch generation.
- No full Python type inference.
- Rankings are heuristic and benchmark-driven.
- Current realistic benchmarks are intentionally small.
- Embedding behavior may vary by model and version.

## Roadmap

### Current / v0.1

- Retrieval-first pytest diagnosis.
- Persistent and incremental local indexing.
- Semantic, dependency-aware, and call-path-aware retrieval.
- Deterministic ScoreBreakdown-backed explanations.
- Realistic benchmark apps.
- Expected root cause found in the top 3 on the current benchmark set.

### Next

- Benchmark evaluator/report command.
- Structured assertion diff extraction.
- Larger benchmark set with more bug patterns.
- Optional LLM diagnosis over retrieved context.
- VS Code integration later.
- Patch suggestions much later.

## Testing

```bash
python -m pytest
ruff check .
python -m pytest examples/realistic_bugs
```

`examples/realistic_bugs` intentionally contains failing tests; the expected current result is `3 failed, 9 passed`.
