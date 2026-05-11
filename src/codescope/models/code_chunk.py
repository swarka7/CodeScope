from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CodeChunk:
    """Represents a chunk of code for later embedding and retrieval."""

    file_path: Path
    start_line: int
    end_line: int
    content: str

