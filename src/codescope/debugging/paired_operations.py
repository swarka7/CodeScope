from __future__ import annotations

import re
from dataclasses import dataclass

from codescope.debugging.failure_signals import (
    FailureSignals,
    identifier_tokens,
)
from codescope.models.code_chunk import CodeChunk

PAIRED_OPERATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("add", "remove"),
    ("approve", "reject"),
    ("attach", "detach"),
    ("close", "open"),
    ("credit", "debit"),
    ("deposit", "withdraw"),
    ("disable", "enable"),
    ("exclude", "include"),
    ("grant", "revoke"),
    ("lock", "unlock"),
    ("reserve", "release"),
    ("start", "stop"),
)

PAIRED_OPERATION_WORDS = frozenset(
    operation for pair in PAIRED_OPERATION_PAIRS for operation in pair
)

STATE_CONTEXT_WORDS = frozenset(
    {
        "activity",
        "amount",
        "balance",
        "count",
        "counterpart",
        "destination",
        "inventory",
        "item",
        "money",
        "movement",
        "order",
        "quantity",
        "record",
        "source",
        "state",
        "status",
        "stock",
        "target",
        "total",
    }
)

_OPERATION_COUNTERPARTS: dict[str, tuple[str, ...]] = {}
for left, right in PAIRED_OPERATION_PAIRS:
    _OPERATION_COUNTERPARTS.setdefault(left, tuple())
    _OPERATION_COUNTERPARTS.setdefault(right, tuple())
    _OPERATION_COUNTERPARTS[left] = tuple(sorted({*_OPERATION_COUNTERPARTS[left], right}))
    _OPERATION_COUNTERPARTS[right] = tuple(sorted({*_OPERATION_COUNTERPARTS[right], left}))

_CALL_RE = re.compile(r"(?:\.|\b)(?P<name>[A-Za-z_]\w*)\s*\(")


@dataclass(frozen=True, slots=True)
class PairedOperationEvidence:
    called_terms: tuple[str, ...]
    counterpart_terms: tuple[str, ...]
    details: tuple[str, ...]

    @property
    def has_evidence(self) -> bool:
        return bool(self.called_terms and self.counterpart_terms)


def paired_operation_counterparts(term: str) -> tuple[str, ...]:
    return _OPERATION_COUNTERPARTS.get(term.lower(), tuple())


def has_paired_state_failure_context(
    *, signals: FailureSignals, primary_terms: set[str]
) -> bool:
    terms = (
        set(signals.operation_words)
        | set(signals.behavioral_words)
        | set(signals.domain_words)
        | primary_terms
    )
    return bool(
        terms
        & (
            PAIRED_OPERATION_WORDS
            | STATE_CONTEXT_WORDS
            | {
                "change",
                "changed",
                "move",
                "moved",
                "transfer",
                "transferred",
                "update",
                "updated",
            }
        )
    )


def paired_operation_evidence(
    *, chunk: CodeChunk, signals: FailureSignals, primary_terms: set[str]
) -> PairedOperationEvidence:
    if signals.did_not_raise:
        return PairedOperationEvidence((), (), ())

    if not has_paired_state_failure_context(signals=signals, primary_terms=primary_terms):
        return PairedOperationEvidence((), (), ())

    called_terms = called_paired_operation_terms(chunk)
    if not called_terms:
        return PairedOperationEvidence((), (), ())

    counterpart_terms = _counterparts_for_terms(called_terms)
    if not counterpart_terms:
        return PairedOperationEvidence((), (), ())

    details = tuple(
        f"{term}->{counterpart}"
        for term in called_terms
        for counterpart in counterpart_terms
        if counterpart in paired_operation_counterparts(term)
    )
    return PairedOperationEvidence(
        called_terms=called_terms,
        counterpart_terms=counterpart_terms,
        details=details,
    )


def called_paired_operation_terms(chunk: CodeChunk) -> tuple[str, ...]:
    terms: list[str] = []
    for dependency in chunk.dependencies:
        terms.extend(_paired_terms_from_identifier(dependency.rsplit(".", 1)[-1]))

    for line in chunk.source_code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            continue
        for match in _CALL_RE.finditer(line):
            terms.extend(_paired_terms_from_identifier(match.group("name")))

    return _dedupe(terms)


def chunk_defines_operation(chunk: CodeChunk, operation: str) -> bool:
    if chunk.chunk_type not in {"function", "method"}:
        return False
    return _name_starts_with_operation(chunk.name, operation)


def chunk_defines_paired_operation(chunk: CodeChunk) -> bool:
    return any(chunk_defines_operation(chunk, operation) for operation in PAIRED_OPERATION_WORDS)


def counterpart_terms_for_called_operations(chunk: CodeChunk) -> tuple[str, ...]:
    return _counterparts_for_terms(called_paired_operation_terms(chunk))


def _counterparts_for_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
    counterparts: list[str] = []
    for term in terms:
        counterparts.extend(paired_operation_counterparts(term))
    return _dedupe(counterparts)


def _paired_terms_from_identifier(value: str) -> list[str]:
    return [
        token
        for token in identifier_tokens(value)
        if token in PAIRED_OPERATION_WORDS
    ]


def _name_starts_with_operation(name: str, operation: str) -> bool:
    normalized = name.lower().lstrip("_")
    return (
        normalized == operation
        or normalized.startswith(f"{operation}_")
        or normalized.startswith(operation)
    )


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return tuple(result)
