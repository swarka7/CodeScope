"""Structured failure signals used by deterministic failure-aware ranking.

Business-logic failures often name the user-facing action in the test
(`archive_task`, `transfer`, `authorize`) while the most diagnostic source lives
in guard or validation code (`validate_status_transition`, `validate_daily_limit`).
`DID NOT RAISE` is especially important: it means the operation was allowed when
the test expected a rejection path, so chunks that raise/check/validate should
receive additional ranking weight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure


@dataclass(frozen=True, slots=True)
class FailureSignals:
    did_not_raise: bool
    expected_exceptions: tuple[str, ...]
    exception_symbols: tuple[str, ...]
    behavioral_words: tuple[str, ...]
    operation_words: tuple[str, ...]
    domain_words: tuple[str, ...]


VALIDATION_NAME_PREFIXES = (
    "validate",
    "check",
    "ensure",
    "guard",
    "can",
    "allow",
    "authorize",
)

VALIDATION_WORDS = frozenset(
    {
        "allow",
        "authorize",
        "can",
        "check",
        "deny",
        "error",
        "exception",
        "forbidden",
        "guard",
        "invalid",
        "raise",
        "reject",
        "unauthorized",
        "validate",
        "validation",
        "validator",
        "validators",
    }
)

_BEHAVIORAL_WORDS = frozenset(
    {
        "archived",
        "cannot",
        "deny",
        "denied",
        "expired",
        "forbidden",
        "invalid",
        "limit",
        "reject",
        "rejected",
        "should",
        "threshold",
        "transition",
        "unauthorized",
        "validation",
    }
)

_OPERATION_WORDS = frozenset(
    {
        "accept",
        "accepted",
        "allow",
        "allowed",
        "archive",
        "archived",
        "approve",
        "approved",
        "attach",
        "attached",
        "authorize",
        "balance",
        "cancel",
        "cancelled",
        "charge",
        "close",
        "closed",
        "credit",
        "create",
        "created",
        "debit",
        "delete",
        "deleted",
        "deposit",
        "detach",
        "detached",
        "disable",
        "disabled",
        "enable",
        "enabled",
        "exclude",
        "excluded",
        "filter",
        "filtered",
        "grant",
        "granted",
        "include",
        "included",
        "list",
        "lock",
        "locked",
        "mark",
        "marked",
        "move",
        "moved",
        "open",
        "opened",
        "query",
        "rank",
        "record",
        "recorded",
        "reject",
        "rejected",
        "remove",
        "removed",
        "reserve",
        "reserved",
        "revoke",
        "revoked",
        "search",
        "ship",
        "shipped",
        "sort",
        "start",
        "started",
        "stop",
        "stopped",
        "transfer",
        "transferred",
        "unlock",
        "unlocked",
        "update",
        "updated",
        "validate",
        "withdraw",
        "withdrawn",
    }
)

_STOP_WORDS = frozenset(
    {
        "and",
        "assert",
        "class",
        "did",
        "does",
        "error",
        "expected",
        "failed",
        "false",
        "for",
        "from",
        "got",
        "is",
        "not",
        "none",
        "pytest",
        "raise",
        "raises",
        "should",
        "test",
        "tests",
        "the",
        "true",
        "with",
    }
)

_DID_NOT_RAISE_RE = re.compile(
    r"DID NOT RAISE\s+<class\s+'(?P<qualified>[^']+)'>", re.IGNORECASE
)
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_WORD_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+")
_NORMALIZED_WORDS = {
        "accepted": "accept",
        "allowed": "allow",
        "approved": "approve",
        "attached": "attach",
        "balances": "balance",
        "closed": "close",
        "created": "create",
        "deleted": "delete",
        "denied": "deny",
        "detached": "detach",
        "disabled": "disable",
        "enabled": "enable",
        "excluded": "exclude",
        "filters": "filter",
        "filtered": "filter",
        "granted": "grant",
        "included": "include",
        "locked": "lock",
        "marked": "mark",
        "moves": "move",
        "moved": "move",
        "opened": "open",
        "records": "record",
        "recorded": "record",
        "rejected": "reject",
        "released": "release",
        "removed": "remove",
        "revoked": "revoke",
        "reserves": "reserve",
        "reserved": "reserve",
        "ships": "ship",
        "shipped": "ship",
        "shipping": "ship",
        "started": "start",
        "stopped": "stop",
        "transferred": "transfer",
        "unlocked": "unlock",
        "updated": "update",
        "withdrawn": "withdraw",
    }


def extract_failure_signals(failure: TestFailure) -> FailureSignals:
    text = "\n".join(
        [
            failure.test_name,
            failure.error_type or "",
            failure.message,
            failure.traceback,
        ]
    )

    expected_exceptions = _extract_expected_exceptions(text)
    exception_symbols = _extract_exception_symbols(expected_exceptions)
    tokens = _tokens_with_variants(text)

    behavioral_words = _select_terms(tokens, _BEHAVIORAL_WORDS)
    operation_words = _select_terms(tokens, _OPERATION_WORDS)
    domain_words = _domain_words(tokens)

    return FailureSignals(
        did_not_raise=bool(_DID_NOT_RAISE_RE.search(text)),
        expected_exceptions=tuple(expected_exceptions),
        exception_symbols=tuple(exception_symbols),
        behavioral_words=tuple(behavioral_words),
        operation_words=tuple(operation_words),
        domain_words=tuple(domain_words),
    )


def chunk_signal_text(chunk: CodeChunk) -> str:
    return "\n".join(
        [
            chunk.name,
            chunk.parent or "",
            chunk.source_code,
            *chunk.dependencies,
            *chunk.decorators,
        ]
    )


def chunk_signal_tokens(chunk: CodeChunk) -> set[str]:
    return set(_tokens_with_variants(chunk_signal_text(chunk)))


def has_validation_name(name: str) -> bool:
    lower = name.lower().lstrip("_")
    for prefix in VALIDATION_NAME_PREFIXES:
        if lower == prefix or lower.startswith(f"{prefix}_"):
            return True
        if prefix != "can" and lower.startswith(prefix):
            return True
    return False


def contains_raise_statement(source: str) -> bool:
    return re.search(r"\braise\b", source, re.IGNORECASE) is not None


def relevant_exception_symbol_matches(tokens: set[str], signals: FailureSignals) -> set[str]:
    generic_exception_words = {"error", "exception", "invalid"}
    exception_name_terms = expected_exception_short_name_tokens(signals)
    return (exception_name_terms & tokens) - generic_exception_words


def defines_expected_exception(chunk: CodeChunk, signals: FailureSignals) -> bool:
    if chunk.chunk_type != "class":
        return False
    return chunk.name in expected_exception_short_names(signals)


def contains_expected_exception(text: str, signals: FailureSignals) -> bool:
    text_lower = text.lower()
    return any(exception.lower() in text_lower for exception in signals.expected_exceptions)


def raises_expected_exception(source: str, signals: FailureSignals) -> bool:
    for exception_name in expected_exception_short_names(signals):
        pattern = rf"\braise\s+(?:[A-Za-z_]\w*\.)*{re.escape(exception_name)}\b"
        if re.search(pattern, source):
            return True
    return False


def expected_exception_short_names(signals: FailureSignals) -> set[str]:
    return {exception.rsplit(".", 1)[-1] for exception in signals.expected_exceptions}


def expected_exception_short_name_tokens(signals: FailureSignals) -> set[str]:
    tokens: set[str] = set()
    for exception_name in expected_exception_short_names(signals):
        tokens.update(identifier_tokens(exception_name))
    return tokens


def calls_validation_helper(chunk: CodeChunk) -> bool:
    for dependency in chunk.dependencies:
        dependency_name = dependency.rsplit(".", 1)[-1]
        if has_validation_name(dependency_name):
            return True

    for prefix in VALIDATION_NAME_PREFIXES:
        if prefix == "can":
            pattern = r"\bcan_[A-Za-z0-9_]*\s*\("
        else:
            pattern = rf"\b{re.escape(prefix)}[A-Za-z0-9_]*\s*\("
        for line in chunk.source_code.splitlines():
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                continue
            if re.search(pattern, line):
                return True

    return False


def calls_relevant_validation_helper(chunk: CodeChunk, signals: FailureSignals) -> bool:
    relevant_terms = expected_exception_short_name_tokens(signals) - {
        "error",
        "exception",
        "invalid",
    }

    for dependency in chunk.dependencies:
        dependency_name = dependency.rsplit(".", 1)[-1]
        dependency_tokens = identifier_tokens(dependency_name)
        if has_validation_name(dependency_name) and dependency_tokens & relevant_terms:
            return True

    for call_match in re.finditer(r"\b(?P<name>[A-Za-z_]\w*)\s*\(", chunk.source_code):
        call_name = call_match.group("name")
        call_tokens = identifier_tokens(call_name)
        if has_validation_name(call_name) and call_tokens & relevant_terms:
            return True

    return False


def identifier_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _WORD_RE.findall(value):
        for part in raw.split("_"):
            if not part:
                continue
            pieces = _CAMEL_WORD_RE.findall(part) or [part]
            tokens.update(piece.lower() for piece in pieces if piece)
    return tokens


def _extract_expected_exceptions(text: str) -> list[str]:
    names: list[str] = []
    for match in _DID_NOT_RAISE_RE.finditer(text):
        qualified = match.group("qualified").strip()
        short_name = qualified.rsplit(".", 1)[-1]
        names.extend([short_name, qualified])
    return _dedupe(names)


def _extract_exception_symbols(expected_exceptions: list[str]) -> list[str]:
    symbols: list[str] = []
    for exception in expected_exceptions:
        symbols.extend(_tokens_with_variants(exception))
    return _dedupe(symbols)


def _tokens_with_variants(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _tokens(text):
        tokens.append(token)
        normalized = _NORMALIZED_WORDS.get(token)
        if normalized:
            tokens.append(normalized)
    return _dedupe(tokens)


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _WORD_RE.findall(text):
        for part in raw.replace(".", "_").split("_"):
            if not part:
                continue
            camel_parts = _CAMEL_WORD_RE.findall(part) or [part]
            tokens.extend(piece.lower() for piece in camel_parts if piece)
    return tokens


def _select_terms(tokens: list[str], vocabulary: frozenset[str]) -> list[str]:
    return _dedupe([token for token in tokens if token in vocabulary])


def _domain_words(tokens: list[str]) -> list[str]:
    words: list[str] = []
    for token in tokens:
        if len(token) <= 2:
            continue
        if token in _STOP_WORDS:
            continue
        words.append(token)
    return _dedupe(words)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
