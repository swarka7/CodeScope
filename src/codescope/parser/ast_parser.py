from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ParsedPythonFile:
    file_path: Path
    source: str
    module: ast.Module | None
    error: SyntaxError | OSError | UnicodeError | None = None


class AstParser:
    """Reads and parses Python files into an AST.

    Syntax errors are captured and returned so the caller can continue processing
    other files without failing the whole pipeline.
    """

    def parse_file(self, file_path: Path) -> ParsedPythonFile:
        file_path = Path(file_path)

        try:
            source = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                source = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                return ParsedPythonFile(file_path=file_path, source="", module=None, error=exc)
        except OSError as exc:
            return ParsedPythonFile(file_path=file_path, source="", module=None, error=exc)

        try:
            module = ast.parse(source, filename=str(file_path))
        except SyntaxError as exc:
            return ParsedPythonFile(file_path=file_path, source=source, module=None, error=exc)

        return ParsedPythonFile(file_path=file_path, source=source, module=module, error=None)
