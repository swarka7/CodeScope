# Failure-Aware Diagnosis

CodeScope’s `diagnose` command connects pytest failures to likely relevant source code from an existing local index. It does not generate patches, apply fixes, or claim a proven root cause.

Use `diagnose` when you want CodeScope to run pytest and analyze real failure output. Use `investigate` when you have a natural-language bug description and want likely relevant code to inspect without running tests.

## Pipeline

```text
pytest output
  ↓
Failure parser
  ↓
Failure query builder
  ↓
Semantic retrieval
  ↓
Failure-aware reranking
  ↓
ScoreBreakdown-backed scoring
  ↓
Dependency-aware context
  ↓
Call-path + reverse-call context
  ↓
Diagnosis summary + retrieval reasons
```

## What CodeScope extracts from failures

- Failed test name and file location
- Error type and concise failure message
- Traceback symbols and source file hints
- Expected exception names from pytest messages such as `DID NOT RAISE`
- Behavioral words such as `invalid`, `expired`, `unauthorized`, `limit`, and `transition`
- Operation words such as `update`, `transfer`, `delete`, `authorize`, and `validate`

## Retrieval reasons

Each diagnose result can include deterministic reasons such as:

- `defines expected exception`
- `references expected exception`
- `raises expected exception`
- `validation logic`
- `calls validation logic`
- `traceback file match`
- `keyword match: ...`
- `operation match: ...`

These reasons are rule-based and are meant to make the ranking auditable. They are not LLM-generated explanations.

The user-facing reasons are backed by structured scoring components where possible. That keeps ranking evidence and explanations aligned instead of maintaining two unrelated sets of heuristics.

## Ranking Signals

Failure-aware ranking is deterministic and static. It can use:

- Source-first ranking to keep test chunks visible but prioritize source code.
- Expected-exception evidence such as definitions, references, and `raise` statements.
- Call-path context from source chunks to validators, guards, helpers, and exception logic.
- Reverse-call context from validators or exception chunks back to likely business callers.
- Business-behavior signals such as filtering logic, state updates, and operation-name matches.
- Paired state-operation reasoning for missing counterpart patterns such as debit/credit-style flows.

These signals are intentionally heuristic. They improve ranking quality, but they do not prove the root cause.

## Example

```text
CodeScope Diagnose

Status
- Tests failed

Failing test
- [FAIL] tests/test_auth_service.py::test_expired_token_is_rejected

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

## JSON output

`diagnose --json` is intended for tools and future editor integrations:

```bash
python -m codescope.cli diagnose <repo> --json
```

JSON mode preserves the same deterministic retrieval behavior and exit codes as human-readable diagnose. It writes a single JSON object to stdout containing failures, diagnosis summaries, likely relevant code, related context, scores, reasons, and optional LLM status when `--llm` is also enabled.

For natural-language bug descriptions, `investigate --json` has the same JSON-only stdout contract:

```bash
python -m codescope.cli investigate <repo> "bug description" --json
```

When `investigate --json --llm` is used, CodeScope adds a top-level `llm` object. Its `status` is `completed`, `skipped`, or `error`.

## Design constraints

CodeScope intentionally prioritizes:

- Retrieval quality first
- Explainability first
- Deterministic behavior first
- Generalized heuristics before LLM reasoning

Patch generation and automated repair are planned future milestones, but the current system focuses on finding and explaining the most useful debugging context.

Current benchmark outcomes are documented in [`benchmark_results.md`](benchmark_results.md).

## Optional LLM reasoning

`diagnose --llm` adds an optional AI-generated section after the normal deterministic diagnose output. `investigate --llm` does the same for natural-language bug descriptions. The LLM does not replace CodeScope retrieval:

1. CodeScope runs deterministic retrieval first.
2. For `diagnose`, CodeScope builds a compact context packet from the failure summary, retrieved chunks, reasons, dependencies, and bounded snippets.
3. For `investigate`, CodeScope builds a compact context packet from the bug description, retrieved chunks, reasons, dependencies, and bounded snippets.
3. The configured provider receives that context.
4. Output appears under a clearly labeled `LLM Diagnosis` or `LLM Investigation` section.

The `fake` provider is useful for testing the pipeline without network access or an API key.

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

Fake-provider investigation example:

```powershell
$env:CODESCOPE_LLM_PROVIDER="fake"
python -m codescope.cli investigate examples/realistic_bugs/banking_app "When I transfer money, the receiver balance does not increase" --llm
Remove-Item Env:CODESCOPE_LLM_PROVIDER
```

Fake-provider JSON investigation example:

```powershell
$env:CODESCOPE_LLM_PROVIDER="fake"
python -m codescope.cli investigate examples/realistic_bugs/banking_app "When I transfer money, the receiver balance does not increase" --json --llm
Remove-Item Env:CODESCOPE_LLM_PROVIDER
```

To use the optional OpenAI provider, install the extra:

```bash
python -m pip install -e ".[openai]"
```

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

The same OpenAI configuration works for `investigate --llm` and `investigate --json --llm`.

Trust boundaries:

- `--llm` is optional.
- The output is AI-generated reasoning over retrieved CodeScope context.
- The OpenAI provider sends retrieved/redacted CodeScope context to OpenAI.
- Real provider usage requires internet access and may incur API cost.
- Missing provider configuration does not break normal diagnose or investigate.
- CodeScope does not modify files or generate patches.
- Deterministic CodeScope output remains the source of truth.

## Limitations

- Ranking is heuristic and static.
- Traceback-to-source mapping is useful but not complete.
- Dependency traversal is intentionally bounded to avoid noisy result explosions.
- Retrieval reasons explain why a chunk was selected, not whether it is definitely the root cause.
