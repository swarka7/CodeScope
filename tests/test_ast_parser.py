from __future__ import annotations

from pathlib import Path

from codescope.parser.ast_parser import AstParser


def test_parses_valid_python_file(tmp_path: Path) -> None:
    file_path = tmp_path / "valid.py"
    file_path.write_text("def foo():\n    return 1\n", encoding="utf-8")

    parsed = AstParser().parse_file(file_path)

    assert parsed.module is not None
    assert parsed.error is None
    assert "def foo()" in parsed.source


def test_handles_syntax_errors_gracefully(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid.py"
    file_path.write_text("def foo(:\n    return 1\n", encoding="utf-8")

    parsed = AstParser().parse_file(file_path)

    assert parsed.module is None
    assert isinstance(parsed.error, SyntaxError)
    assert "def foo(" in parsed.source

