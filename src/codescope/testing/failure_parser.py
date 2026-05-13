from __future__ import annotations

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
            traceback_text = _extract_traceback(lines, display_name=display_name, location=location)

            message = summary.message.strip()
            if not message:
                message = _extract_message_from_traceback(traceback_text)

            failures.append(
                TestFailure(
                    test_name=summary.nodeid,
                    file_path=file_path,
                    line_number=location.line_number if location else None,
                    error_type=location.error_type if location else None,
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


def _extract_message_from_traceback(traceback: str) -> str:
    for raw in traceback.splitlines():
        line = raw.strip()
        if not line.startswith("E"):
            continue

        # Normalize both `E assert ...` and `E AssertionError: ...` forms.
        message = line.removeprefix("E").strip()
        if message:
            return message

    return ""

