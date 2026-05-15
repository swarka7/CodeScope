from __future__ import annotations

from pathlib import Path

from codescope.parser.ast_parser import AstParser
from codescope.parser.chunker import Chunker


def test_extracts_functions_classes_methods_and_imports(tmp_path: Path) -> None:
    source = "\n".join(
        [
            "import os",
            "from typing import List, Dict as D",
            "",
            "def top_level(a: int) -> int:",
            "    def nested() -> int:",
            "        return 0",
            "    return a",
            "",
            "class MyClass:",
            "    def method(self) -> int:",
            "        return 1",
            "",
            "    async def async_method(self) -> int:",
            "        return 2",
            "",
            "def another():",
            "    pass",
            "",
        ]
    )
    file_path = tmp_path / "sample.py"
    file_path.write_text(source, encoding="utf-8")

    parsed = AstParser().parse_file(file_path)
    chunks = Chunker().extract_chunks(parsed)

    imports = ["import os", "from typing import List, Dict as D"]

    function_names = sorted([c.name for c in chunks if c.chunk_type == "function"])
    assert function_names == ["another", "top_level"]

    class_names = [c.name for c in chunks if c.chunk_type == "class"]
    assert class_names == ["MyClass"]

    method_names = sorted([f"{c.parent}.{c.name}" for c in chunks if c.chunk_type == "method"])
    assert method_names == ["MyClass.async_method", "MyClass.method"]

    for chunk in chunks:
        assert chunk.file_path == file_path.as_posix()
        assert chunk.imports == imports
        assert chunk.dependencies == []
        expected_source = "".join(
            source.splitlines(keepends=True)[chunk.start_line - 1 : chunk.end_line]
        )
        assert chunk.source_code == expected_source


def test_returns_empty_for_unparsable_files(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid.py"
    file_path.write_text("def broken(:\n    pass\n", encoding="utf-8")

    parsed = AstParser().parse_file(file_path)
    chunks = Chunker().extract_chunks(parsed)

    assert chunks == []


def test_extracts_simple_decorators(tmp_path: Path) -> None:
    source = "\n".join(
        [
            "from dataclasses import dataclass",
            "",
            "@dataclass",
            "class User:",
            "    name: str",
            "",
            "class Factory:",
            "    @classmethod",
            "    def build(cls) -> User:",
            "        return User(name='Ada')",
            "",
            "    @staticmethod",
            "    def ping() -> bool:",
            "        return True",
            "",
        ]
    )
    file_path = tmp_path / "sample.py"
    file_path.write_text(source, encoding="utf-8")

    parsed = AstParser().parse_file(file_path)
    chunks = Chunker().extract_chunks(parsed)

    user = next(c for c in chunks if c.chunk_type == "class" and c.name == "User")
    build = next(c for c in chunks if c.chunk_type == "method" and c.name == "build")
    ping = next(c for c in chunks if c.chunk_type == "method" and c.name == "ping")

    assert user.decorators == ["@dataclass"]
    assert build.decorators == ["@classmethod"]
    assert ping.decorators == ["@staticmethod"]


def test_extracts_fastapi_route_decorators(tmp_path: Path) -> None:
    source = "\n".join(
        [
            "from fastapi import FastAPI",
            "",
            "app = FastAPI()",
            "",
            '@app.get("/")',
            "def list_todos() -> list[str]:",
            "    return []",
            "",
            '@app.post("/add")',
            "def create_todo() -> dict[str, str]:",
            "    return {'ok': 'true'}",
            "",
            '@router.delete(\"/{id}\")',
            "def delete_todo(id: int) -> None:",
            "    return None",
            "",
        ]
    )
    file_path = tmp_path / "routes.py"
    file_path.write_text(source, encoding="utf-8")

    parsed = AstParser().parse_file(file_path)
    chunks = Chunker().extract_chunks(parsed)

    by_name = {chunk.name: chunk for chunk in chunks}
    assert by_name["list_todos"].decorators == ['@app.get("/")']
    assert by_name["create_todo"].decorators == ['@app.post("/add")']
    assert by_name["delete_todo"].decorators == ['@router.delete("/{id}")']
