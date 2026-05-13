from __future__ import annotations

from pathlib import Path

import pytest

import codescope.cli as cli_module
from codescope.cli import main as cli_main
from codescope.debugging.failure_retriever import FailureRetriever
from codescope.embeddings.embedder import Embedder
from codescope.indexing.index_store import IndexStore
from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure


def test_failure_query_construction() -> None:
    failure = TestFailure(
        test_name="tests/test_auth.py::test_invalid_token",
        file_path="tests/test_auth.py",
        line_number=42,
        error_type="AssertionError",
        message="expected 401 but got 200",
        traceback="tests/test_auth.py:42: AssertionError",
    )

    query = FailureRetriever.build_failure_query(failure)

    assert "Test:" in query
    assert "tests/test_auth.py::test_invalid_token" in query
    assert "Error:" in query
    assert "AssertionError" in query
    assert "Message:" in query
    assert "expected 401 but got 200" in query
    assert "Traceback symbols:" in query
    assert "Source hints:" in query
    assert "tests/test_auth.py:42" in query


def test_failure_query_truncates_long_traceback_excerpt() -> None:
    long_line = "x" * 200
    traceback = "\n".join([f"{i}:{long_line}" for i in range(50)])

    failure = TestFailure(
        test_name="tests/test_example.py::test_fail",
        file_path="tests/test_example.py",
        line_number=10,
        error_type="AssertionError",
        message="boom",
        traceback=traceback,
    )

    query = FailureRetriever.build_failure_query(failure)

    assert "Traceback excerpt:" in query
    assert query.strip().endswith("...")


def test_failure_query_extracts_source_hints_and_symbols_and_removes_duplicates() -> None:
    traceback = "\n".join(
        [
            'File "src/auth/service.py", line 22, in validate_token',
            "    return decode(token)",
            'File "src/auth/service.py", line 22, in validate_token',
            "    return decode(token)",
            ">   assert validate_token('bad') is True",
            "tests/test_auth.py:42: AssertionError",
            "tests/test_auth.py:42: AssertionError",
        ]
    )

    failure = TestFailure(
        test_name="tests/test_auth.py::test_invalid_token",
        file_path="tests/test_auth.py",
        line_number=42,
        error_type="AssertionError",
        message="",
        traceback=traceback,
    )

    query = FailureRetriever.build_failure_query(failure)

    assert "src/auth/service.py:22" in query
    assert query.count("src/auth/service.py:22") == 1

    assert "tests/test_auth.py:42" in query
    assert query.count("tests/test_auth.py:42") == 1

    assert "validate_token" in query
    assert query.lower().count("validate_token") >= 1


def test_failure_query_without_traceback_still_contains_core_context() -> None:
    failure = TestFailure(
        test_name="tests/test_auth.py::test_invalid_token",
        file_path="tests/test_auth.py",
        line_number=42,
        error_type="AssertionError",
        message="expected 401 but got 200",
        traceback="",
    )

    query = FailureRetriever.build_failure_query(failure)

    assert "tests/test_auth.py::test_invalid_token" in query
    assert "AssertionError" in query
    assert "expected 401 but got 200" in query
    assert "tests/test_auth.py:42" in query


def test_diagnose_prints_tests_passed_when_no_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_path = tmp_path / "repo"
    _write_pytest_project(
        repo_path,
        files={},
        test_source="\n".join(
            [
                "def test_ok() -> None:",
                "    assert 1 + 1 == 2",
                "",
            ]
        ),
    )

    exit_code = cli_main(["diagnose", str(repo_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "Tests passed"


def test_diagnose_requires_index_when_tests_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_path = tmp_path / "repo"
    _write_pytest_project(
        repo_path,
        files={},
        test_source="\n".join(
            [
                "def test_fail() -> None:",
                "    assert False",
                "",
            ]
        ),
    )

    exit_code = cli_main(["diagnose", str(repo_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.err.strip() == (
        "No CodeScope index found. Run: python -m codescope.cli index <repo_path>"
    )


def test_failure_aware_retrieval_returns_relevant_chunks(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    auth_file = repo_path / "auth_service.py"
    jwt_file = repo_path / "jwt.py"
    auth_file.write_text("def validate_token(token: str) -> bool:\n    return False\n", "utf-8")
    jwt_file.write_text("class JWTManager:\n    pass\n", "utf-8")

    chunks = [
        _chunk(
            id="validate",
            file_path=auth_file.as_posix(),
            chunk_type="function",
            name="validate_token",
            dependencies=["JWTManager"],
            start_line=1,
            end_line=2,
            source_code="def validate_token(token: str) -> bool:\n    return False\n",
        ),
        _chunk(
            id="jwt",
            file_path=jwt_file.as_posix(),
            chunk_type="class",
            name="JWTManager",
            dependencies=[],
            start_line=1,
            end_line=2,
            source_code="class JWTManager:\n    pass\n",
        ),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    IndexStore(repo_path).save(
        chunks=chunks,
        embeddings=embeddings,
        metadata={"schema_version": 1, "chunks_indexed": len(chunks), "files_indexed": 2},
    )

    failure = TestFailure(
        test_name="tests/test_auth.py::test_invalid_token",
        file_path="tests/test_auth.py",
        line_number=None,
        error_type="AssertionError",
        message="assert validate_token('bad') is True",
        traceback=">   assert validate_token('bad') is True",
    )

    retriever = FailureRetriever(repo_path, embedder=Embedder(model=_KeywordModel()))
    results = retriever.retrieve(failure, top_k=1)

    assert [r.kind for r in results] == ["semantic", "related"]
    assert [r.chunk.name for r in results] == ["validate_token", "JWTManager"]


def test_diagnose_outputs_failure_summary_and_likely_relevant_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_path = tmp_path / "repo"
    _write_pytest_project(
        repo_path,
        files={
            "auth_service.py": "\n".join(
                [
                    "def validate_token(token: str) -> bool:",
                    "    return False",
                    "",
                ]
            )
        },
        test_source="\n".join(
            [
                "from auth_service import validate_token",
                "",
                "def test_invalid_token() -> None:",
                "    assert validate_token('bad') is True",
                "",
            ]
        ),
    )

    auth_file = repo_path / "auth_service.py"
    jwt_file = repo_path / "jwt.py"
    jwt_file.write_text("class JWTManager:\n    pass\n", "utf-8")

    chunks = [
        _chunk(
            id="validate",
            file_path=auth_file.as_posix(),
            chunk_type="function",
            name="validate_token",
            dependencies=["JWTManager"],
            start_line=1,
            end_line=2,
            source_code=auth_file.read_text(encoding="utf-8"),
        ),
        _chunk(
            id="jwt",
            file_path=jwt_file.as_posix(),
            chunk_type="class",
            name="JWTManager",
            dependencies=[],
            start_line=1,
            end_line=2,
            source_code=jwt_file.read_text(encoding="utf-8"),
        ),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    IndexStore(repo_path).save(
        chunks=chunks,
        embeddings=embeddings,
        metadata={"schema_version": 1, "chunks_indexed": len(chunks), "files_indexed": 2},
    )

    class _PatchedFailureRetriever(FailureRetriever):
        def __init__(self, repo: Path) -> None:
            super().__init__(repo, embedder=Embedder(model=_KeywordModel()))

    monkeypatch.setattr(cli_module, "FailureRetriever", _PatchedFailureRetriever)

    exit_code = cli_main(["diagnose", str(repo_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Tests failed" in captured.out
    assert "[FAIL] tests/test_example.py::test_invalid_token" in captured.out
    assert "Likely relevant code:" in captured.out
    assert "validate_token" in captured.out


class _KeywordModel:
    def encode_query(self, text: str, *, normalize_embeddings: bool) -> list[float]:
        _ = normalize_embeddings
        lower = text.lower()
        if "validate_token" in lower:
            return [1.0, 0.0, 0.0]
        if "jwtmanager" in lower or "jwt" in lower:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def _chunk(
    *,
    id: str,
    file_path: str,
    chunk_type: str,
    name: str,
    dependencies: list[str],
    start_line: int,
    end_line: int,
    source_code: str,
) -> CodeChunk:
    return CodeChunk(
        id=id,
        file_path=file_path,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        name=name,
        parent=None,
        start_line=start_line,
        end_line=end_line,
        source_code=source_code,
        imports=[],
        dependencies=dependencies,
    )


def _write_pytest_project(
    repo_path: Path, *, files: dict[str, str], test_source: str, test_file: str = "test_example.py"
) -> None:
    repo_path.mkdir(parents=True, exist_ok=True)

    for rel, content in files.items():
        target = repo_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content + "\n", encoding="utf-8")

    tests_dir = repo_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / test_file).write_text(test_source + "\n", encoding="utf-8")
