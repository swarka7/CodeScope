from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class CodeChunk:
    """A structural chunk of Python code extracted from a source file."""

    id: str
    file_path: str
    chunk_type: Literal["module", "class", "function", "method"]
    name: str
    parent: str | None
    start_line: int
    end_line: int
    source_code: str
    imports: list[str]
    dependencies: list[str]
    decorators: list[str] = field(default_factory=list)
