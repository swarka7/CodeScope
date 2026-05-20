from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import codescope.cli as cli_module
from codescope.benchmark import (
    BenchmarkCase,
    BenchmarkEvaluation,
    BenchmarkResult,
    classify_rank,
    evaluate_benchmarks,
)
from codescope.models.code_chunk import CodeChunk
from codescope.retrieval.dependency_aware import RetrievalResult


def test_classify_rank_uses_benchmark_thresholds() -> None:
    assert classify_rank(1) == "PASS"
    assert classify_rank(3) == "PASS"
    assert classify_rank(4) == "PARTIAL"
    assert classify_rank(5) == "PARTIAL"
    assert classify_rank(6) == "FAIL"
    assert classify_rank(None) == "FAIL"


def test_evaluate_benchmarks_reports_pass_partial_and_fail(tmp_path: Path) -> None:
    root = tmp_path / "benchmarks"
    root.mkdir()
    for name in ("app_pass", "app_partial", "app_fail"):
        (root / name).mkdir()

    indexed_paths: list[Path] = []

    evaluation = evaluate_benchmarks(
        root,
        cases=(
            BenchmarkCase("app_pass", Path("app_pass"), "Service.fix"),
            BenchmarkCase("app_partial", Path("app_partial"), "Service.fix"),
            BenchmarkCase("app_fail", Path("app_fail"), "Service.fix"),
        ),
        indexer_factory=lambda path: _FakeIndexer(path, indexed_paths),
        test_runner=_FakeRunner(),
        retriever_factory=lambda path: _FakeRetriever(path.name),
    )

    assert [path.name for path in indexed_paths] == ["app_pass", "app_partial", "app_fail"]
    assert [
        (result.name, result.observed_rank, result.result) for result in evaluation.results
    ] == [
        ("app_pass", 1, "PASS"),
        ("app_partial", 4, "PARTIAL"),
        ("app_fail", None, "FAIL"),
    ]
    assert evaluation.pass_count == 1
    assert evaluation.partial_count == 1
    assert evaluation.fail_count == 1
    assert not evaluation.successful


def test_evaluate_benchmarks_raises_for_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        evaluate_benchmarks(tmp_path / "missing")


def test_benchmark_cli_prints_table_and_returns_zero_for_all_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli_module,
        "evaluate_benchmarks",
        lambda path: BenchmarkEvaluation(
            results=(
                BenchmarkResult("banking_app", "TransferService.transfer", 1, "PASS"),
                BenchmarkResult("movie_platform", "MovieSearchService.search", 1, "PASS"),
            )
        ),
    )

    exit_code = cli_module.main(["benchmark", "examples/realistic_bugs"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "CodeScope Benchmark Report" in captured.out
    assert "benchmark" in captured.out
    assert "expected root cause" in captured.out
    assert "banking_app" in captured.out
    assert "TransferService.transfer" in captured.out
    assert "2 PASS, 0 PARTIAL, 0 FAIL" in captured.out


def test_benchmark_cli_returns_one_for_partial_or_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli_module,
        "evaluate_benchmarks",
        lambda path: BenchmarkEvaluation(
            results=(
                BenchmarkResult("app_partial", "Service.fix", 4, "PARTIAL"),
                BenchmarkResult("app_fail", "Service.fix", None, "FAIL"),
            )
        ),
    )

    exit_code = cli_module.main(["benchmark", "examples/realistic_bugs"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "0 PASS, 1 PARTIAL, 1 FAIL" in captured.out


class _FakeIndexer:
    def __init__(self, path: Path, indexed_paths: list[Path]) -> None:
        self._path = path
        self._indexed_paths = indexed_paths

    def index(self) -> None:
        self._indexed_paths.append(self._path)


class _FakeRunner:
    def run(self, repo_path: Path) -> _RunResult:
        _ = repo_path
        return _RunResult(
            stdout="FAILED tests/test_app.py::test_business_rule - AssertionError",
            stderr="",
            exit_code=1,
        )


@dataclass(frozen=True, slots=True)
class _RunResult:
    stdout: str
    stderr: str
    exit_code: int


class _FakeRetriever:
    def __init__(self, app_name: str) -> None:
        self._app_name = app_name

    def retrieve(self, failure: object, *, top_k: int = 5) -> list[RetrievalResult]:
        _ = failure
        _ = top_k
        if self._app_name == "app_pass":
            names = ["fix", "other", "third"]
        elif self._app_name == "app_partial":
            names = ["first", "second", "third", "fix", "fifth"]
        else:
            names = ["first", "second", "third", "fourth", "fifth"]

        return [
            RetrievalResult(
                kind="semantic",
                chunk=_chunk(name=name, parent="Service" if name == "fix" else None),
                score=1.0,
            )
            for name in names
        ]


def _chunk(name: str, parent: str | None = None) -> CodeChunk:
    return CodeChunk(
        id=f"{parent or 'module'}-{name}",
        file_path="app/service.py",
        chunk_type="method" if parent else "function",
        name=name,
        parent=parent,
        start_line=1,
        end_line=2,
        source_code=f"def {name}():\n    pass\n",
        imports=[],
        dependencies=[],
    )
