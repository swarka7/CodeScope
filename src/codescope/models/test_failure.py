from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TestFailure:
    __test__ = False

    test_name: str
    file_path: str
    line_number: int | None
    error_type: str | None
    message: str
    traceback: str
