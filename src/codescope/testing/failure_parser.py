from __future__ import annotations

import re
from dataclasses import dataclass

from codescope.models.test_failure import TestFailure


@dataclass(frozen=True, slots=True)
class _SummaryEntry:
    kind: str
    nodeid: str
    message: str


@dataclass(frozen=True, slots=True)
class _LocationInfo:
    line_index: int
    file_path: str
    line_number: int | None
    error_type: str | None


class FailureParser:
    """Extract structured failure data from pytest terminal output."""

    def parse(self, output: str) -> list[TestFailure]:
        lines = output.splitlines()
        summaries = _extract_summary_entries(lines)
        if not summaries:
            return []

        failures: list[TestFailure] = []
        for summary in summaries:
            file_path = summary.nodeid.split("::", 1)[0]
            display_name = _display_test_name(summary.nodeid)

            location = _find_location(lines, file_path=file_path)
            error_type = location.error_type if location else None
            traceback_text = _extract_traceback(lines, display_name=display_name, location=location)

            message = summary.message.strip()
            extracted_message = _extract_message_from_traceback(
                traceback_text, error_type=error_type
            )

            if error_type == "AssertionError":
                # Summary messages are frequently truncated; prefer the detailed traceback message.
                if extracted_message:
                    message = extracted_message
            elif not message:
                message = extracted_message

            failures.append(
                TestFailure(
                    test_name=summary.nodeid,
                    file_path=file_path,
                    line_number=location.line_number if location else None,
                    error_type=error_type,
                    message=message,
                    traceback=traceback_text,
                )
            )

        return failures


def _extract_summary_entries(lines: list[str]) -> list[_SummaryEntry]:
    entries: list[_SummaryEntry] = []
    for raw in lines:
        line = raw.strip()
        if line.startswith("FAILED "):
            kind = "FAILED"
            rest = line.removeprefix("FAILED ").strip()
        elif line.startswith("ERROR "):
            kind = "ERROR"
            rest = line.removeprefix("ERROR ").strip()
        else:
            continue

        nodeid, message = _split_nodeid_and_message(rest)
        if nodeid:
            entries.append(_SummaryEntry(kind=kind, nodeid=nodeid, message=message))

    seen: set[tuple[str, str]] = set()
    unique: list[_SummaryEntry] = []
    for entry in entries:
        key = (entry.kind, entry.nodeid)
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def _split_nodeid_and_message(rest: str) -> tuple[str, str]:
    if " - " in rest:
        left, right = rest.split(" - ", 1)
        return left.strip(), right.strip()
    return rest.strip(), ""


def _display_test_name(nodeid: str) -> str:
    last = nodeid.rsplit("::", 1)[-1]
    if "[" in last:
        last = last.split("[", 1)[0]
    return last


def _find_location(lines: list[str], *, file_path: str) -> _LocationInfo | None:
    normalized_target = _normalize_path(file_path)
    for idx, line in enumerate(lines):
        parsed = _parse_location_line(line)
        if parsed is None:
            continue

        normalized_candidate = _normalize_path(parsed.file_path)
        if normalized_candidate == normalized_target:
            return _LocationInfo(
                line_index=idx,
                file_path=parsed.file_path,
                line_number=parsed.line_number,
                error_type=parsed.error_type,
            )

        if normalized_candidate.endswith("/" + normalized_target):
            return _LocationInfo(
                line_index=idx,
                file_path=parsed.file_path,
                line_number=parsed.line_number,
                error_type=parsed.error_type,
            )

    return None


@dataclass(frozen=True, slots=True)
class _ParsedLocationLine:
    file_path: str
    line_number: int
    error_type: str


def _parse_location_line(line: str) -> _ParsedLocationLine | None:
    # Example: tests/test_example.py:14: AssertionError
    parts = line.strip().rsplit(":", 2)
    if len(parts) != 3:
        return None

    file_part, line_part, rest = parts
    file_part = file_part.strip()
    line_part = line_part.strip()
    rest = rest.strip()
    if not file_part or not line_part.isdigit() or not rest:
        return None

    error_type = rest.split(None, 1)[0].strip()
    if not error_type:
        return None

    return _ParsedLocationLine(
        file_path=file_part,
        line_number=int(line_part),
        error_type=error_type,
    )


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized.removeprefix("./")
    return normalized.strip("/").lower()


def _extract_traceback(
    lines: list[str], *, display_name: str, location: _LocationInfo | None
) -> str:
    if location is None:
        return "\n".join(lines).strip()

    start = location.line_index
    for i in range(location.line_index, -1, -1):
        candidate = lines[i].strip()
        if _is_failure_header(candidate) and display_name in candidate:
            start = i
            break

    end = len(lines)
    for i in range(location.line_index + 1, len(lines)):
        candidate = lines[i].strip()
        if _is_section_separator(candidate):
            end = i
            break
        if _is_failure_header(candidate):
            end = i
            break

    return "\n".join(lines[start:end]).strip()


def _is_failure_header(line: str) -> bool:
    if not line.startswith("_"):
        return False
    if not line.endswith("_"):
        return False
    if len(line) < 10:
        return False
    # A header line contains more than just underscores.
    return line.strip("_").strip() != ""


def _is_section_separator(line: str) -> bool:
    if len(line) < 10:
        return False
    if not (line.startswith("=") and line.endswith("=")):
        return False
    return True


_WHERE_LINE_RE = re.compile(r"""^\+\s*where\s+(?P<value>.+?)\s*=\s*(?P<expr>.+)$""")
_ASSERT_EQUALS_RE = re.compile(r"""^assert\s+(?P<left>.+?)\s*==\s*(?P<right>.+)$""")


def _extract_message_from_traceback(traceback: str, *, error_type: str | None) -> str:
    if error_type == "AssertionError":
        return _extract_assertion_message(traceback)
    return _extract_first_error_line(traceback)


def _extract_first_error_line(traceback: str) -> str:
    for raw in traceback.splitlines():
        line = raw.strip()
        if not line or _is_failure_header(line) or _is_section_separator(line):
            continue
        if not line.startswith("E"):
            continue

        message = line.removeprefix("E").strip()
        if not message:
            continue

        return _truncate_one_line(message)

    return ""


def _extract_assertion_message(traceback: str) -> str:
    """Extract a concise assertion message from pytest's failure traceback."""
    where_map: dict[str, str] = {}
    candidates: list[str] = []
    source_assert: str | None = None

    for raw in traceback.splitlines():
        line = raw.strip()
        if not line or _is_failure_header(line) or _is_section_separator(line):
            continue

        if line.startswith(">") and source_assert is None and "assert " in line:
            # Example: `>       assert eval(test_input) == expected`
            source_assert = line.split("assert ", 1)[1].strip()

        if not line.startswith("E"):
            continue

        payload = line.removeprefix("E").strip()
        if not payload:
            continue

        where_match = _WHERE_LINE_RE.match(payload)
        if where_match:
            where_map[where_match.group("value").strip()] = where_match.group("expr").strip()
            continue

        # Skip separator-ish or diff marker lines.
        if payload in {"+", "-", "?", "|"}:
            continue

        candidates.append(payload)

    best = _pick_best_assertion_candidate(candidates)
    if not best and source_assert:
        best = f"assert {source_assert}"

    if not best:
        return _extract_first_error_line(traceback)

    best = _strip_assertionerror_prefix(best)

    equals_match = _ASSERT_EQUALS_RE.match(best)
    if equals_match:
        left = equals_match.group("left").strip()
        right = equals_match.group("right").strip()
        expr = where_map.get(left)
        if expr and "(" in expr:
            best = f"{expr} returned {left} instead of {right}"

    return _truncate_one_line(best)


def _pick_best_assertion_candidate(candidates: list[str]) -> str:
    # Prefer the standard pytest assertion summary line, in both forms:
    # - `AssertionError: assert 3 == 4`
    # - `assert 3 == 4`
    for candidate in candidates:
        if candidate.startswith("AssertionError:") and "assert " in candidate:
            return candidate
    for candidate in candidates:
        if candidate.startswith("assert "):
            return candidate
    for candidate in candidates:
        if candidate.startswith("AssertionError:"):
            return candidate
    return candidates[0] if candidates else ""


def _strip_assertionerror_prefix(message: str) -> str:
    if not message.startswith("AssertionError:"):
        return message
    stripped = message.removeprefix("AssertionError:").strip()
    return stripped or message


def _truncate_one_line(message: str, *, max_chars: int = 220) -> str:
    single_line = " ".join(message.strip().split())
    if len(single_line) <= max_chars:
        return single_line
    return single_line[: max_chars - 3].rstrip() + "..."
