# Demo: CodeScope Search, Investigate, and Diagnose

This demo shows CodeScope indexing a realistic benchmark repository, running semantic search, investigating a natural-language bug description, diagnosing a failing pytest case, and optionally exercising the `--llm` flow.

CodeScope retrieves likely relevant debugging context. It does **not** generate or apply patches.

## Prerequisites

Install the project with development and AI dependencies:

```bash
python -m pip install -e ".[dev,ai]"
```

Install the optional OpenAI provider only if you want real model-backed `--llm` output:

```bash
python -m pip install -e ".[openai]"
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

## 3. Investigate a natural-language bug description

Use `investigate` when you have a bug description but do not want CodeScope to run pytest:

```bash
python -m codescope.cli investigate examples/realistic_bugs/banking_app "When I transfer money, the receiver balance does not increase"
```

Expected output shape:

```text
CodeScope Investigate

Query:
When I transfer money, the receiver balance does not increase

Likely relevant code:
1. TransferService.transfer
   Kind: method
   Location: app/service.py:...
   Source: semantic
   Score: ...
   reasons=
     - semantic match
     - operation match: balance, receiver, transfer
     - business operation
     - state update logic

Related context:
1. Account.credit
   Kind: method
   Location: app/models.py:...
   Source: related
   reasons=
     - paired state operation
     - possible missing counterpart operation
```

For tools and future editor integrations, use JSON output:

```bash
python -m codescope.cli investigate examples/realistic_bugs/banking_app "When I transfer money, the receiver balance does not increase" --json
```

`investigate --json` writes one valid JSON object to stdout and does not include human-readable section headers.

Optional fake-provider LLM investigation:

Windows PowerShell:

```powershell
$env:CODESCOPE_LLM_PROVIDER="fake"
python -m codescope.cli investigate examples/realistic_bugs/banking_app "When I transfer money, the receiver balance does not increase" --llm
Remove-Item Env:CODESCOPE_LLM_PROVIDER
```

JSON with optional fake-provider LLM investigation:

```powershell
$env:CODESCOPE_LLM_PROVIDER="fake"
python -m codescope.cli investigate examples/realistic_bugs/banking_app "When I transfer money, the receiver balance does not increase" --json --llm
Remove-Item Env:CODESCOPE_LLM_PROVIDER
```

`investigate --json --llm` still writes JSON-only stdout and adds a top-level `llm` object with `status` set to `completed`, `skipped`, or `error`.

## 4. Diagnose the failing test

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

For tools and future editor integrations, use JSON output:

```bash
python -m codescope.cli diagnose examples/realistic_bugs/banking_app --json
```

`diagnose --json` writes one valid JSON object to stdout. It preserves the same ranking and exit behavior as normal diagnose.

## 5. Benchmark evaluator

Run all realistic benchmark apps through CodeScope’s benchmark evaluator:

```bash
python -m codescope.cli benchmark examples/realistic_bugs
```

Expected output shape:

```text
CodeScope Benchmark Report

benchmark        expected root cause            rank   result
banking_app      TransferService.transfer       1      PASS
movie_platform   MovieSearchService.search      1      PASS
inventory_app    FulfillmentService.ship_order  3      PASS

Summary:
3 PASS, 0 PARTIAL, 0 FAIL
```

The evaluator checks whether the expected root-cause chunk appears in the top 3 likely relevant code results. It does not fix benchmark bugs.

## 6. Optional LLM pipeline

CodeScope can optionally pass deterministic diagnose or investigate context to a configured LLM provider. Normal deterministic output still prints first, and retrieval remains the source of truth.

The `fake` provider is for testing the pipeline only:

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

For investigation, the human section is labeled `LLM Investigation`:

```text
LLM Investigation
AI-generated reasoning based only on retrieved CodeScope context.

Fake LLM diagnosis based on provided CodeScope context.
```

OpenAI provider:

Windows PowerShell:

```powershell
$env:CODESCOPE_LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="..."
python -m codescope.cli diagnose examples/realistic_bugs/banking_app --llm
Remove-Item Env:CODESCOPE_LLM_PROVIDER
Remove-Item Env:OPENAI_API_KEY
```

Linux/macOS:

```bash
CODESCOPE_LLM_PROVIDER=openai OPENAI_API_KEY="..." python -m codescope.cli diagnose examples/realistic_bugs/banking_app --llm
```

Optional model override:

```bash
CODESCOPE_LLM_PROVIDER=openai CODESCOPE_LLM_MODEL="gpt-5-mini" OPENAI_API_KEY="..." python -m codescope.cli diagnose examples/realistic_bugs/banking_app --llm
```

The same provider settings work with `investigate --llm` and `investigate --json --llm`:

```bash
CODESCOPE_LLM_PROVIDER=openai OPENAI_API_KEY="..." python -m codescope.cli investigate examples/realistic_bugs/banking_app "When I transfer money, the receiver balance does not increase" --llm
```

The OpenAI provider sends retrieved/redacted CodeScope context to OpenAI. It requires internet access, may incur API cost, and produces AI-generated reasoning over the retrieved context. Model output may be wrong. CodeScope still does not modify files or generate patches.

If `--llm` is used without a configured provider, CodeScope still prints normal deterministic output and reports that the LLM section was skipped. In JSON mode, `llm.status` is `completed`, `skipped`, or `error`.

## What the demo proves

- CodeScope can index a realistic multi-file Python example.
- Semantic search can find code by intent, not just exact text.
- Dependency-aware retrieval can include nearby validation and exception context.
- Failure-aware diagnosis can connect a pytest failure to likely source code.
- `investigate` can retrieve likely relevant code from a natural-language bug description without running tests.
- Explanations are deterministic and rule-based.
- JSON output is available for tooling and future editor integrations.
- The optional LLM path can receive bounded retrieved/redacted context without replacing deterministic retrieval.

## What the demo does not do

- It does not fix the bug.
- It does not generate a patch.
- It does not guarantee the LLM explanation is correct.
- It does not prove root cause with runtime analysis.

## Benchmark results

For current v0.1 benchmark outcomes on the realistic banking, movie, and inventory examples, see [`benchmark_results.md`](benchmark_results.md).
