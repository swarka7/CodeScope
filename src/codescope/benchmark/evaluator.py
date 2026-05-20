from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from codescope.indexing.indexer import Indexer
from codescope.models.code_chunk import CodeChunk
from codescope.retrieval.dependency_aware import RetrievalResult
from codescope.testing.failure_parser import FailureParser
from codescope.testing.test_runner import TestRunner, TestRunResult


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    relative_path: Path
    expected_root_cause: str


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    expected_root_cause: str
    observed_rank: int | None
    result: str


@dataclass(frozen=True, slots=True)
class BenchmarkEvaluation:
    results: tuple[BenchmarkResult, ...]

    @property
    def pass_count(self) -> int:
        return _count_results(self.results, "PASS")

    @property
    def partial_count(self) -> int:
        return _count_results(self.results, "PARTIAL")

    @property
    def fail_count(self) -> int:
        return _count_results(self.results, "FAIL")

    @property
    def successful(self) -> bool:
        return self.partial_count == 0 and self.fail_count == 0


DEFAULT_BENCHMARKS: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="banking_app",
        relative_path=Path("banking_app"),
        expected_root_cause="TransferService.transfer",
    ),
    BenchmarkCase(
        name="movie_platform",
        relative_path=Path("movie_platform"),
        expected_root_cause="MovieSearchService.search",
    ),
    BenchmarkCase(
        name="inventory_app",
        relative_path=Path("inventory_app"),
        expected_root_cause="FulfillmentService.ship_order",
    ),
)


def evaluate_benchmarks(
    benchmarks_path: Path,
    *,
    cases: Sequence[BenchmarkCase] = DEFAULT_BENCHMARKS,
    indexer_factory: Callable[[Path], object] = Indexer,
    test_runner: TestRunner | None = None,
    failure_parser: FailureParser | None = None,
    retriever_factory: Callable[[Path], object] | None = None,
) -> BenchmarkEvaluation:
    from codescope.debugging.failure_retriever import FailureRetriever

    root_path = Path(benchmarks_path)
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_path}")

    runner = test_runner or TestRunner()
    parser = failure_parser or FailureParser()
    make_retriever = retriever_factory or FailureRetriever

    results: list[BenchmarkResult] = []
    for case in cases:
        app_path = root_path / case.relative_path
        if not app_path.exists():
            raise FileNotFoundError(f"Benchmark app does not exist: {app_path}")
        if not app_path.is_dir():
            raise NotADirectoryError(f"Benchmark app is not a directory: {app_path}")

        indexer = indexer_factory(app_path)
        index_method = getattr(indexer, "index")
        index_method()

        run_result = runner.run(app_path)
        observed_rank = _observed_root_cause_rank(
            app_path=app_path,
            run_result=run_result,
            parser=parser,
            retriever=make_retriever(app_path),
            expected_root_cause=case.expected_root_cause,
        )
        results.append(
            BenchmarkResult(
                name=case.name,
                expected_root_cause=case.expected_root_cause,
                observed_rank=observed_rank,
                result=classify_rank(observed_rank),
            )
        )

    return BenchmarkEvaluation(results=tuple(results))


def classify_rank(rank: int | None) -> str:
    if rank is None:
        return "FAIL"
    if rank <= 3:
        return "PASS"
    if rank <= 5:
        return "PARTIAL"
    return "FAIL"


def _observed_root_cause_rank(
    *,
    app_path: Path,
    run_result: TestRunResult,
    parser: FailureParser,
    retriever: object,
    expected_root_cause: str,
) -> int | None:
    _ = app_path
    combined_output = "\n".join([run_result.stdout, run_result.stderr]).strip()
    failures = parser.parse(combined_output)
    if not failures:
        return None

    best_rank: int | None = None
    retrieve = getattr(retriever, "retrieve")
    for failure in failures:
        retrieval_results = retrieve(failure, top_k=5)
        rank = _rank_in_semantic_results(retrieval_results, expected_root_cause)
        if rank is None:
            continue
        if best_rank is None or rank < best_rank:
            best_rank = rank
    return best_rank


def _rank_in_semantic_results(
    results: Sequence[RetrievalResult], expected_root_cause: str
) -> int | None:
    rank = 0
    for result in results:
        if result.kind != "semantic":
            continue
        rank += 1
        if rank > 5:
            return None
        if _chunk_display_name(result.chunk) == expected_root_cause:
            return rank
    return None


def _chunk_display_name(chunk: CodeChunk) -> str:
    if chunk.chunk_type == "method" and chunk.parent:
        return f"{chunk.parent}.{chunk.name}"
    return chunk.name


def _count_results(results: Sequence[BenchmarkResult], outcome: str) -> int:
    return sum(1 for result in results if result.result == outcome)
