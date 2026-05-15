from __future__ import annotations

from pathlib import Path

import pytest

import codescope.cli as cli_module
from codescope.cli import main as cli_main
from codescope.debugging.failure_retriever import FailureRetriever
from codescope.embeddings.embedder import Embedder
from codescope.indexing.index_store import IndexStore
from codescope.indexing.index_versions import EMBEDDING_TEXT_VERSION, INDEX_SCHEMA_VERSION
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
        metadata=_index_metadata(chunks_indexed=len(chunks), files_indexed=2),
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


def test_failure_aware_ranking_prefers_source_when_symbol_only_in_traceback(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    test_chunk = _chunk(
        id="test",
        file_path=(repo_path / "tests" / "test_auth_service.py").as_posix(),
        chunk_type="function",
        name="test_expired_token_is_rejected",
        dependencies=[],
        start_line=1,
        end_line=6,
        source_code="\n".join(
            [
                "from auth_service import validate_token",
                "",
                "def test_expired_token_is_rejected() -> None:",
                "    result = validate_token('expired')",
                "    assert result is False",
                "",
            ]
        ),
    )
    source_chunk = _chunk(
        id="src",
        file_path=(repo_path / "auth_service.py").as_posix(),
        chunk_type="function",
        name="validate_token",
        dependencies=[],
        start_line=1,
        end_line=2,
        source_code="def validate_token(token: str) -> bool:\n    return True\n",
    )

    IndexStore(repo_path).save(
        chunks=[test_chunk, source_chunk],
        embeddings=[
            [1.0, 0.0],  # semantic match favors tests first
            [0.7, 0.7],
        ],
        metadata=_index_metadata(chunks_indexed=2, files_indexed=2),
    )

    failure = TestFailure(
        test_name="tests/test_auth_service.py::test_expired_token_is_rejected",
        file_path="tests/test_auth_service.py",
        line_number=5,
        error_type="AssertionError",
        message="assert True is False",
        traceback="\n".join(
            [
                "def test_expired_token_is_rejected() -> None:",
                "    result = validate_token('expired')",
                ">   assert result is False",
                "E   assert True is False",
            ]
        ),
    )

    retriever = FailureRetriever(repo_path, embedder=Embedder(model=_FixedQueryModel([1.0, 0.0])))
    results = retriever.retrieve(failure, top_k=2)

    assert [r.kind for r in results] == ["semantic", "semantic"]
    assert [r.chunk.name for r in results] == ["validate_token", "test_expired_token_is_rejected"]
    assert len({r.chunk.id for r in results}) == len(results)


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
        metadata=_index_metadata(chunks_indexed=len(chunks), files_indexed=2),
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
    assert "Diagnosis summary:" in captured.out
    assert "Likely relevant code:" in captured.out
    assert "validate_token" in captured.out


def test_failure_aware_ranking_prefers_source_over_test_chunk(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    source_file = repo_path / "calculator.py"
    test_file = repo_path / "tests" / "test_calculator.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)

    source_file.write_text(
        "\n".join(
            [
                "def calculate_discount(price: int, percent: int) -> int:",
                "    return price - (price * percent)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    test_file.write_text(
        "\n".join(
            [
                "from calculator import calculate_discount",
                "",
                "def test_calculate_discount_applies_percent() -> None:",
                "    assert calculate_discount(100, 10) == 90",
                "",
            ]
        ),
        encoding="utf-8",
    )

    source_chunk = _chunk(
        id="calc",
        file_path=source_file.as_posix(),
        chunk_type="function",
        name="calculate_discount",
        dependencies=[],
        start_line=1,
        end_line=2,
        source_code=source_file.read_text(encoding="utf-8"),
    )
    test_chunk = _chunk(
        id="test",
        file_path=test_file.as_posix(),
        chunk_type="function",
        name="test_calculate_discount_applies_percent",
        dependencies=["calculate_discount"],
        start_line=1,
        end_line=4,
        source_code=test_file.read_text(encoding="utf-8"),
    )

    # Make the test chunk win on raw cosine similarity (before reranking).
    embeddings = [
        [1.0, 0.0],  # test chunk: similarity 1.0 vs query [1, 0]
        [0.8, 0.6],  # source chunk: similarity 0.8 vs query [1, 0]
    ]

    IndexStore(repo_path).save(
        chunks=[test_chunk, source_chunk],
        embeddings=embeddings,
        metadata=_index_metadata(chunks_indexed=2, files_indexed=2),
    )

    failure = TestFailure(
        test_name="tests/test_calculator.py::test_calculate_discount_applies_percent",
        file_path="tests/test_calculator.py",
        line_number=4,
        error_type="AssertionError",
        message="calculate_discount(100, 10) returned -900 instead of 90",
        traceback=">   assert calculate_discount(100, 10) == 90",
    )

    retriever = FailureRetriever(repo_path, embedder=Embedder(model=_FixedQueryModel([1.0, 0.0])))
    results = retriever.retrieve(failure, top_k=2)

    assert [r.kind for r in results] == ["semantic", "semantic"]
    assert [r.chunk.name for r in results] == [
        "calculate_discount",
        "test_calculate_discount_applies_percent",
    ]
    assert len({r.chunk.id for r in results}) == len(results)


def test_failure_aware_ranking_keeps_tests_when_strongly_relevant(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    test_chunk = _chunk(
        id="test",
        file_path=(repo_path / "tests" / "test_app.py").as_posix(),
        chunk_type="function",
        name="test_app",
        dependencies=[],
        start_line=1,
        end_line=2,
        source_code="def test_app() -> None:\n    assert False\n",
    )
    source_chunk = _chunk(
        id="src",
        file_path=(repo_path / "app.py").as_posix(),
        chunk_type="function",
        name="run",
        dependencies=[],
        start_line=1,
        end_line=2,
        source_code="def run() -> None:\n    pass\n",
    )

    embeddings = [
        [1.0, 0.0],  # test chunk remains the strongest semantic match
        [0.0, 1.0],
    ]
    IndexStore(repo_path).save(
        chunks=[test_chunk, source_chunk],
        embeddings=embeddings,
        metadata=_index_metadata(chunks_indexed=2, files_indexed=2),
    )

    failure = TestFailure(
        test_name="tests/test_app.py::test_app",
        file_path="tests/test_app.py",
        line_number=1,
        error_type="AssertionError",
        message="assert False",
        traceback=">   assert False",
    )

    retriever = FailureRetriever(repo_path, embedder=Embedder(model=_FixedQueryModel([1.0, 0.0])))
    first = retriever.retrieve(failure, top_k=2)
    second = retriever.retrieve(failure, top_k=2)

    assert [r.chunk.id for r in first] == [r.chunk.id for r in second]
    assert [r.chunk.id for r in first] == ["test", "src"]


class _KeywordModel:
    def encode_query(self, text: str, *, normalize_embeddings: bool) -> list[float]:
        _ = normalize_embeddings
        lower = text.lower()
        if "validate_token" in lower:
            return [1.0, 0.0, 0.0]
        if "jwtmanager" in lower or "jwt" in lower:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class _FixedQueryModel:
    def __init__(self, embedding: list[float]) -> None:
        self._embedding = embedding

    def encode_query(self, text: str, *, normalize_embeddings: bool) -> list[float]:
        _ = (text, normalize_embeddings)
        return list(self._embedding)


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


def _index_metadata(*, chunks_indexed: int, files_indexed: int) -> dict[str, object]:
    return {
        "schema_version": 2,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "embedding_model_name": "all-MiniLM-L6-v2",
        "chunks_indexed": chunks_indexed,
        "files_indexed": files_indexed,
    }


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
