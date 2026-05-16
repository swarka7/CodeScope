# Failure-Aware Diagnosis

CodeScope’s `diagnose` command connects pytest failures to likely relevant source code from an existing local index. It does not generate patches, apply fixes, or claim a proven root cause.

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
Dependency-aware context
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

## Design constraints

CodeScope intentionally prioritizes:

- Retrieval quality first
- Explainability first
- Deterministic behavior first
- Generalized heuristics before LLM reasoning

Patch generation and automated repair are planned future milestones, but the current system focuses on finding and explaining the most useful debugging context.

Current benchmark outcomes are documented in [`benchmark_results.md`](benchmark_results.md).

## Limitations

- Ranking is heuristic and static.
- Traceback-to-source mapping is useful but not complete.
- Dependency traversal is intentionally bounded to avoid noisy result explosions.
- Retrieval reasons explain why a chunk was selected, not whether it is definitely the root cause.
