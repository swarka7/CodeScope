# Demo: CodeScope Search and Failure-Aware Diagnosis

This demo shows CodeScope indexing a realistic benchmark repository, running semantic search, diagnosing a failing pytest case, and optionally exercising the fake-provider `--llm` flow.

CodeScope retrieves likely relevant debugging context. It does **not** generate or apply patches.

## Prerequisites

Install the project with development and AI dependencies:

```bash
python -m pip install -e ".[dev,ai]"
```

## 1. Build the local index

Index the realistic banking benchmark:

```bash
python -m codescope.cli index examples/realistic_bugs/banking_app
```

Expected shape:

```text
Indexed N new/changed files
Reused 0 unchanged files
Removed 0 deleted files
Total chunks: N
```

Exact counts may change as the example evolves.

## 2. Run semantic search

Search for transfer and balance logic:

```bash
python -m codescope.cli search examples/realistic_bugs/banking_app "money transfer balance"
```

Expected output includes semantic matches and dependency-aware related context:

```text
[semantic] [method] TransferService.transfer app/service.py:... score=...
[related]  [method] Account.credit app/models.py:...
```

## 3. Diagnose the failing test

Run pytest through CodeScope diagnosis:

```bash
python -m codescope.cli diagnose examples/realistic_bugs/banking_app
```

Expected output shape:

```text
CodeScope Diagnose

Status
- Tests failed

Failing test
- [FAIL] tests/test_transfers.py::test_successful_transfer_moves_money_and_records_activity

Failure signal
- Error: AssertionError
- Message: assert (Decimal('75....00'), 1, True) == (Decimal('75....00'), 1, True)

Diagnosis summary:
- Failing test: test_successful_transfer_moves_money_and_records_activity
- Failure signal: AssertionError, assert (Decimal('75....00'), 1, True) == (Decimal('75....00'), 1, True)
- Most relevant source chunk: TransferService.transfer in app/service.py
- Related context: TransferRecord, BankRepository.get_account, BankRepository.record_transfer
- Why: failure/query symbols overlap with retrieved source/context chunks.

Likely relevant code:
1. TransferService.transfer
   Kind: method
   Location: app/service.py:...
   Source: semantic
   Score: ...
   reasons=
     - operation match: balance, record, transfer
     - business operation
     - state update logic
     - paired state operation

2. Account.credit
   Kind: method
   Location: app/models.py:...
   Source: related
   reasons=
     - paired state operation
     - possible missing counterpart operation
```

Scores can vary slightly depending on embedding behavior, but the important contract is:

- Diagnosis summary is printed.
- A cautious possible issue may be printed when rule-based signals are strong enough.
- Likely relevant code and related context are listed in readable sections.
- Each result includes deterministic `reasons=...` text.

## 4. Optional LLM diagnosis pipeline

CodeScope can optionally pass the deterministic diagnose context to a configured LLM provider. The normal diagnose output still prints first, and retrieval remains the source of truth.

The currently documented provider is `fake`. It is for testing the pipeline only:

- It does not call a real model.
- It does not require an API key.
- It does not modify files or generate patches.

Windows PowerShell:

```powershell
$env:CODESCOPE_LLM_PROVIDER="fake"
python -m codescope.cli diagnose examples/realistic_bugs/banking_app --llm
Remove-Item Env:CODESCOPE_LLM_PROVIDER
```

Linux/macOS:

```bash
CODESCOPE_LLM_PROVIDER=fake python -m codescope.cli diagnose examples/realistic_bugs/banking_app --llm
```

Expected fake-provider section:

```text
LLM Diagnosis
AI-generated reasoning based only on retrieved CodeScope context.

Fake LLM diagnosis based on provided CodeScope context.
```

If `--llm` is used without a configured provider, CodeScope still prints normal diagnose output and reports that the LLM section was skipped.

## What the demo proves

- CodeScope can index a realistic multi-file Python example.
- Semantic search can find code by intent, not just exact text.
- Dependency-aware retrieval can include nearby validation and exception context.
- Failure-aware diagnosis can connect a pytest failure to likely source code.
- Explanations are deterministic and rule-based.
- The optional LLM path can receive bounded retrieved context without replacing deterministic retrieval.

## What the demo does not do

- It does not fix the bug.
- It does not generate a patch.
- It does not call a real LLM provider yet.
- It does not prove root cause with runtime analysis.

## Benchmark results

For current v0.1 benchmark outcomes on the realistic banking, movie, and inventory examples, see [`benchmark_results.md`](benchmark_results.md).
