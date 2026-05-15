# Demo: Failure-Aware Diagnosis

This demo shows CodeScope indexing a realistic example repository, running semantic search, and diagnosing a failing pytest case.

CodeScope retrieves likely relevant debugging context. It does **not** generate or apply patches.

## Prerequisites

Install the project with development and AI dependencies:

```bash
python -m pip install -e ".[dev,ai]"
```

## 1. Build the local index

Index the backend-style auth service example:

```bash
python -m codescope.cli index examples/buggy_auth_service
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

Search for expired-token validation logic:

```bash
python -m codescope.cli search examples/buggy_auth_service "expired token validation"
```

Expected output includes semantic matches and dependency-aware related context:

```text
[semantic] [function] validate_token auth_service.py:... score=...
[related]  [function] decode_token token_manager.py:...
[related]  [class] TokenPayload models.py:...
```

## 3. Diagnose the failing test

Run pytest through CodeScope diagnosis:

```bash
python -m codescope.cli diagnose examples/buggy_auth_service
```

Expected output shape:

```text
Tests failed

[FAIL] tests/test_auth_service.py::test_expired_token_is_rejected
Error: AssertionError
Message: assert True is False

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
[semantic] [function] validate_token auth_service.py:... score=... reasons=validation helper name; behavioral keyword overlap: expired, rejected
[related]  [function] decode_token token_manager.py:... reasons=semantic similarity
[related]  [class] TokenPayload models.py:... reasons=semantic similarity
```

Scores can vary slightly depending on embedding behavior, but the important contract is:

- Diagnosis summary is printed.
- A cautious possible issue may be printed when rule-based signals are strong enough.
- Likely relevant code is listed.
- Each result includes deterministic `reasons=...` text.

## What the demo proves

- CodeScope can index a realistic multi-file Python example.
- Semantic search can find code by intent, not just exact text.
- Dependency-aware retrieval can include nearby validation and exception context.
- Failure-aware diagnosis can connect a pytest failure to likely source code.
- Explanations are deterministic and rule-based.

## What the demo does not do

- It does not fix the bug.
- It does not generate a patch.
- It does not call an LLM.
- It does not prove root cause with runtime analysis.
