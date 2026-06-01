from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TypeVar

DEFAULT_MAX_TOTAL_CONTEXT_CHARS = 12_000
DEFAULT_MAX_FAILURE_MESSAGE_CHARS = 1_200
REDACTION_PLACEHOLDER = "[REDACTED]"

_SENSITIVE_NAME_PATTERN = r"[\w.-]*(?:api[_-]?key|token|password|secret)[\w.-]*"
_QUOTED_SECRET_ASSIGNMENT_RE = re.compile(
    rf"(?i)(?P<prefix>[\"']?\b{_SENSITIVE_NAME_PATTERN}\b[\"']?\s*[:=]\s*)"
    r"(?P<quote>[\"'])(?P<value>.*?)(?P=quote)"
)
_UNQUOTED_SECRET_ASSIGNMENT_RE = re.compile(
    rf"(?i)(?P<prefix>\b{_SENSITIVE_NAME_PATTERN}\b\s*=\s*)"
    r"(?P<value>[^\s,;)}\]]+)"
)
_AUTHORIZATION_BEARER_RE = re.compile(
    r"(?i)(authorization\s*:\s*bearer\s+)(?P<value>[A-Za-z0-9._~+/=-]+)"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)(\bbearer\s+)(?P<value>[A-Za-z0-9._~+/=-]+)")

T = TypeVar("T")


def redact_sensitive_text(value: str) -> str:
    if not value:
        return value

    redacted = _QUOTED_SECRET_ASSIGNMENT_RE.sub(_replace_quoted_secret, value)
    redacted = _UNQUOTED_SECRET_ASSIGNMENT_RE.sub(_replace_unquoted_secret, redacted)
    redacted = _AUTHORIZATION_BEARER_RE.sub(
        lambda match: f"{match.group(1)}{REDACTION_PLACEHOLDER}",
        redacted,
    )
    return _BEARER_TOKEN_RE.sub(
        lambda match: f"{match.group(1)}{REDACTION_PLACEHOLDER}",
        redacted,
    )


def truncate_text(value: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return "." * max_chars
    return value[: max_chars - 3].rstrip() + "..."


def truncate_code(source: str, *, max_chars: int, max_lines: int) -> str:
    if max_lines <= 0 or max_chars <= 0:
        return ""

    redacted_source = redact_sensitive_text(source)
    lines = redacted_source.rstrip().splitlines()
    truncated_by_line = len(lines) > max_lines
    snippet = "\n".join(lines[:max_lines])
    if truncated_by_line:
        snippet = snippet.rstrip() + "\n..."

    return truncate_text(snippet, max_chars=max_chars)


def fit_items_to_context_cap(
    items: Sequence[T],
    *,
    fixed_size: int,
    max_chars: int,
    item_size: Callable[[T], int],
) -> tuple[T, ...]:
    """Drop trailing items until the deterministic context size fits the cap."""

    if max_chars <= fixed_size:
        return ()

    selected = list(items)
    while selected and fixed_size + sum(item_size(item) for item in selected) > max_chars:
        selected.pop()
    return tuple(selected)


def _replace_quoted_secret(match: re.Match[str]) -> str:
    quote = match.group("quote")
    return f"{match.group('prefix')}{quote}{REDACTION_PLACEHOLDER}{quote}"


def _replace_unquoted_secret(match: re.Match[str]) -> str:
    return f"{match.group('prefix')}{REDACTION_PLACEHOLDER}"
