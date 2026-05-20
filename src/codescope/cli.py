from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from codescope.benchmark import BenchmarkEvaluation, evaluate_benchmarks
from codescope.debugging.diagnosis_summary import build_diagnosis_summary
from codescope.debugging.failure_retriever import FailureRetriever
from codescope.debugging.issue_hypothesis import build_issue_hypothesis
from codescope.debugging.llm_context import build_llm_diagnosis_context
from codescope.debugging.llm_prompt import build_llm_diagnosis_prompt
from codescope.debugging.retrieval_reasons import build_retrieval_reasons
from codescope.embeddings.embedder import Embedder
from codescope.graph.dependency_graph import DependencyGraph
from codescope.indexing.index_compatibility import check_index_compatibility
from codescope.indexing.index_store import IndexStore
from codescope.indexing.indexer import Indexer
from codescope.llm import LLMProvider, LLMRequest, load_llm_config, load_llm_provider
from codescope.models.code_chunk import CodeChunk
from codescope.models.test_failure import TestFailure
from codescope.parser.ast_parser import AstParser
from codescope.parser.chunker import Chunker
from codescope.retrieval.dependency_aware import RetrievalResult, enrich_with_related
from codescope.scanner.repo_scanner import RepoScanner
from codescope.testing.failure_parser import FailureParser
from codescope.testing.test_runner import TestRunner
from codescope.vectorstore.memory_store import MemoryStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codescope")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan a repository for Python files")
    scan_parser.add_argument("repo_path", type=Path, help="Path to the repository root")

    index_parser = subparsers.add_parser("index", help="Build a local CodeScope index")
    index_parser.add_argument("repo_path", type=Path, help="Path to the repository root")

    chunks_parser = subparsers.add_parser(
        "chunks", help="Extract structural code chunks from a repository"
    )
    chunks_parser.add_argument("repo_path", type=Path, help="Path to the repository root")

    search_parser = subparsers.add_parser("search", help="Semantic search over extracted chunks")
    search_parser.add_argument("repo_path", type=Path, help="Path to the repository root")
    search_parser.add_argument("query", type=str, help="Search query text")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    test_parser = subparsers.add_parser("test", help="Run pytest and extract failure details")
    test_parser.add_argument("repo_path", type=Path, help="Path to the repository root")
    test_parser.add_argument(
        "--test-path",
        type=Path,
        default=None,
        help="Optional path to a test file or directory within the repository",
    )

    diagnose_parser = subparsers.add_parser(
        "diagnose", help="Run tests and retrieve likely relevant code"
    )
    diagnose_parser.add_argument("repo_path", type=Path, help="Path to the repository root")
    diagnose_parser.add_argument(
        "--llm",
        action="store_true",
        help="Add optional AI-generated reasoning over retrieved CodeScope context",
    )
    diagnose_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print machine-readable diagnose results as JSON",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Evaluate CodeScope against benchmark apps"
    )
    benchmark_parser.add_argument(
        "benchmarks_path",
        type=Path,
        help="Path to a directory containing CodeScope benchmark apps",
    )

    return parser


def _handle_scan(repo_path: Path) -> int:
    scanner = RepoScanner()
    try:
        files = scanner.scan(repo_path)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Discovered {len(files)} Python files")
    for file_path in files:
        try:
            display_path = file_path.relative_to(repo_path)
        except ValueError:
            display_path = file_path
        print(display_path)

    return 0


def _handle_index(repo_path: Path) -> int:
    try:
        summary = Indexer(repo_path).index()
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if summary.rebuilt_full_index:
        print("Existing index is outdated; rebuilding full index.")
    print(f"Indexed {summary.indexed_files} new/changed files")
    print(f"Reused {summary.reused_files} unchanged files")
    print(f"Removed {summary.removed_files} deleted files")
    print(f"Total chunks: {summary.total_chunks}")
    return 0


def _handle_chunks(repo_path: Path) -> int:
    scanner = RepoScanner()
    try:
        files = scanner.scan(repo_path)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    parser = AstParser()
    chunker = Chunker()

    total_chunks = 0
    for file_path in files:
        parsed = parser.parse_file(file_path)
        if parsed.module is None:
            continue

        for chunk in chunker.extract_chunks(parsed):
            total_chunks += 1
            if chunk.chunk_type == "method" and chunk.parent:
                chunk_name = f"{chunk.parent}.{chunk.name}"
            else:
                chunk_name = chunk.name
            try:
                display_path = file_path.relative_to(repo_path).as_posix()
            except ValueError:
                display_path = file_path.as_posix()

            location = f"{display_path}:{chunk.start_line}-{chunk.end_line}"
            print(f"[{chunk.chunk_type}] {chunk_name} {location}")

    if total_chunks == 0:
        print("No chunks found")

    return 0


def _handle_search(repo_path: Path, query: str, top_k: int) -> int:
    index_store = IndexStore(repo_path)
    embedder = Embedder()

    try:
        compatibility = check_index_compatibility(
            index_store=index_store,
            embedding_model_name=embedder.model_name,
        )
        if not compatibility.compatible:
            print(compatibility.message, file=sys.stderr)
            return 2
        chunks, embeddings, _metadata = index_store.load()
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        query_embedding = embedder.embed_text(query)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    store = MemoryStore()
    store.add(chunks, embeddings)
    semantic_results = store.search(query_embedding, top_k=top_k)
    graph = DependencyGraph(chunks)
    results = enrich_with_related(query=query, semantic_results=semantic_results, graph=graph)

    for result in results:
        chunk = result.chunk
        if chunk.chunk_type == "method" and chunk.parent:
            chunk_name = f"{chunk.parent}.{chunk.name}"
        else:
            chunk_name = chunk.name

        try:
            display_path = Path(chunk.file_path).relative_to(repo_path).as_posix()
        except ValueError:
            display_path = chunk.file_path

        location = f"{display_path}:{chunk.start_line}-{chunk.end_line}"
        kind_tag = f"[{result.kind}]".ljust(10)
        type_tag = f"[{chunk.chunk_type}]"
        if result.kind == "semantic" and result.score is not None:
            print(f"{kind_tag} {type_tag} {chunk_name} {location} score={result.score:.2f}")
        else:
            print(f"{kind_tag} {type_tag} {chunk_name} {location}")

    return 0


def _handle_test(repo_path: Path, test_path: Path | None) -> int:
    runner = TestRunner()
    run_result = runner.run(repo_path, test_path)
    combined_output = "\n".join([run_result.stdout, run_result.stderr]).strip()

    if run_result.exit_code == 0:
        print("Tests passed")
        return 0

    print("Tests failed")
    failures = FailureParser().parse(combined_output)
    for failure in failures:
        print(f"[FAIL] {failure.test_name}")
        location = failure.file_path
        if failure.line_number is not None:
            location = f"{location}:{failure.line_number}"
        print(f"File: {location}")
        if failure.error_type:
            print(f"Error: {failure.error_type}")
        if failure.message:
            print(f"Message: {failure.message}")

    if not failures and combined_output:
        print(combined_output)

    return run_result.exit_code


def _handle_benchmark(benchmarks_path: Path) -> int:
    try:
        evaluation = evaluate_benchmarks(benchmarks_path)
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    _print_benchmark_report(evaluation)
    return 0 if evaluation.successful else 1


def _print_benchmark_report(evaluation: BenchmarkEvaluation) -> None:
    rows = [
        (
            result.name,
            result.expected_root_cause,
            str(result.observed_rank) if result.observed_rank is not None else "-",
            result.result,
        )
        for result in evaluation.results
    ]
    headers = ("benchmark", "expected root cause", "rank", "result")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]

    print("CodeScope Benchmark Report")
    print()
    print(_format_benchmark_row(headers, widths))
    for row in rows:
        print(_format_benchmark_row(row, widths))
    print()
    print("Summary:")
    print(
        f"{evaluation.pass_count} PASS, "
        f"{evaluation.partial_count} PARTIAL, "
        f"{evaluation.fail_count} FAIL"
    )


def _format_benchmark_row(row: tuple[str, str, str, str], widths: list[int]) -> str:
    return (
        f"{row[0].ljust(widths[0])}  "
        f"{row[1].ljust(widths[1])}  "
        f"{row[2].ljust(widths[2])}  "
        f"{row[3].ljust(widths[3])}"
    )


def _handle_diagnose(
    repo_path: Path, *, use_llm: bool = False, json_output: bool = False
) -> int:
    runner = TestRunner()
    run_result = runner.run(repo_path)

    combined_output = "\n".join([run_result.stdout, run_result.stderr]).strip()

    if json_output:
        return _handle_diagnose_json(
            repo_path=repo_path,
            use_llm=use_llm,
            run_exit_code=run_result.exit_code,
            combined_output=combined_output,
        )

    if run_result.exit_code == 0:
        _print_diagnose_status("Tests passed")
        return 0

    _print_diagnose_status("Tests failed")
    print()

    failures = FailureParser().parse(combined_output)
    if not failures:
        if combined_output:
            print(combined_output)
        return run_result.exit_code

    index_store = IndexStore(repo_path)
    try:
        compatibility = check_index_compatibility(
            index_store=index_store,
            embedding_model_name=Embedder().model_name,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if compatibility.reason == "missing":
        print(compatibility.message, file=sys.stderr)
        return 2

    retriever = FailureRetriever(repo_path)
    llm_provider: LLMProvider | None = None
    llm_provider_error: str | None = None
    llm_config = None
    if use_llm:
        try:
            llm_config = load_llm_config()
            llm_provider = load_llm_provider(llm_config)
        except ValueError as exc:
            llm_provider_error = str(exc)

    for failure in failures:
        _print_failure_details(failure)

        if not compatibility.compatible:
            print(compatibility.message, file=sys.stderr)
            return 2

        try:
            results = retriever.retrieve(failure, top_k=5)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        diagnosis_summary = build_diagnosis_summary(failure, results)
        print(diagnosis_summary)
        hypothesis = build_issue_hypothesis(failure, results)
        if hypothesis:
            print()
            print(hypothesis)
        print()
        _print_retrieval_sections(results, repo_path=repo_path, failure=failure)
        if use_llm:
            print()
            _print_llm_diagnosis(
                failure=failure,
                diagnosis_summary=diagnosis_summary,
                possible_issue=hypothesis,
                results=results,
                provider=llm_provider,
                model=llm_config.model if llm_config is not None else None,
                provider_error=llm_provider_error,
            )
        print()

    return run_result.exit_code


def _handle_diagnose_json(
    *,
    repo_path: Path,
    use_llm: bool,
    run_exit_code: int,
    combined_output: str,
) -> int:
    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        exit_code, payload = _build_diagnose_json_payload(
            repo_path=repo_path,
            use_llm=use_llm,
            run_exit_code=run_exit_code,
            combined_output=combined_output,
        )

    captured = captured_stdout.getvalue()
    if captured:
        print(captured, file=sys.stderr, end="")

    _print_json(payload)
    return exit_code


def _build_diagnose_json_payload(
    *,
    repo_path: Path,
    use_llm: bool,
    run_exit_code: int,
    combined_output: str,
) -> tuple[int, dict[str, object]]:
    if run_exit_code == 0:
        return 0, {
            "schema_version": 1,
            "status": "passed",
            "repo": repo_path.as_posix(),
            "diagnose_exit_code": 0,
            "failures": [],
        }

    failures = FailureParser().parse(combined_output)
    if not failures:
        return run_exit_code, {
            "schema_version": 1,
            "status": "failed",
            "repo": repo_path.as_posix(),
            "diagnose_exit_code": run_exit_code,
            "message": combined_output,
            "failures": [],
        }

    try:
        compatibility = check_index_compatibility(
            index_store=IndexStore(repo_path),
            embedding_model_name=Embedder().model_name,
        )
    except (OSError, ValueError) as exc:
        return 2, _json_error_payload(repo_path=repo_path, message=f"Error: {exc}", exit_code=2)

    if not compatibility.compatible:
        return 2, _json_error_payload(
            repo_path=repo_path, message=compatibility.message, exit_code=2
        )

    retriever = FailureRetriever(repo_path)
    llm_provider: LLMProvider | None = None
    llm_provider_error: str | None = None
    llm_config = None
    if use_llm:
        try:
            llm_config = load_llm_config()
            llm_provider = load_llm_provider(llm_config)
        except ValueError as exc:
            llm_provider_error = str(exc)

    failure_items = []
    for failure in failures:
        try:
            results = retriever.retrieve(failure, top_k=5)
        except ValueError as exc:
            return 2, _json_error_payload(repo_path=repo_path, message=str(exc), exit_code=2)

        diagnosis_summary = build_diagnosis_summary(failure, results)
        possible_issue = build_issue_hypothesis(failure, results)
        failure_items.append(
            _diagnose_failure_json(
                failure=failure,
                results=results,
                repo_path=repo_path,
                diagnosis_summary=diagnosis_summary,
                possible_issue=possible_issue,
                llm=(
                    _llm_diagnosis_json(
                        failure=failure,
                        diagnosis_summary=diagnosis_summary,
                        possible_issue=possible_issue,
                        results=results,
                        provider=llm_provider,
                        model=llm_config.model if llm_config is not None else None,
                        provider_error=llm_provider_error,
                    )
                    if use_llm
                    else None
                ),
            )
        )

    return run_exit_code, {
        "schema_version": 1,
        "status": "failed",
        "repo": repo_path.as_posix(),
        "diagnose_exit_code": run_exit_code,
        "failures": failure_items,
    }


def _json_error_payload(*, repo_path: Path, message: str, exit_code: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "error",
        "repo": repo_path.as_posix(),
        "diagnose_exit_code": exit_code,
        "message": message,
        "failures": [],
    }


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _diagnose_failure_json(
    *,
    failure: TestFailure,
    results: list[RetrievalResult],
    repo_path: Path,
    diagnosis_summary: str,
    possible_issue: str | None,
    llm: dict[str, object] | None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "test_name": failure.test_name,
        "file_path": failure.file_path,
        "line_number": failure.line_number,
        "error_type": failure.error_type,
        "message": failure.message,
        "diagnosis_summary": diagnosis_summary,
        "possible_issue": possible_issue,
        "likely_relevant_code": [
            _retrieval_result_json(rank, result, repo_path=repo_path, failure=failure)
            for rank, result in enumerate(
                [result for result in results if result.kind == "semantic"],
                start=1,
            )
        ],
        "related_context": [
            _retrieval_result_json(rank, result, repo_path=repo_path, failure=failure)
            for rank, result in enumerate(
                [result for result in results if result.kind == "related"],
                start=1,
            )
        ],
    }
    if llm is not None:
        item["llm"] = llm
    return item


def _retrieval_result_json(
    rank: int, result: RetrievalResult, *, repo_path: Path, failure: TestFailure
) -> dict[str, object]:
    chunk = result.chunk
    return {
        "rank": rank,
        "name": _chunk_display_name(chunk),
        "kind": chunk.chunk_type,
        "file_path": _chunk_file_path(chunk, repo_path=repo_path),
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "source": result.kind,
        "score": result.score,
        "reasons": build_retrieval_reasons(failure, chunk, extra_reasons=result.reasons),
        "dependencies": list(chunk.dependencies),
    }


def _llm_diagnosis_json(
    *,
    failure: TestFailure,
    diagnosis_summary: str,
    possible_issue: str | None,
    results: list[RetrievalResult],
    provider: LLMProvider | None,
    model: str | None,
    provider_error: str | None,
) -> dict[str, object]:
    if provider_error:
        return {
            "enabled": True,
            "provider": None,
            "model": model,
            "status": "error",
            "message": "LLM provider unavailable",
        }

    if provider is None:
        return {
            "enabled": True,
            "provider": None,
            "model": model,
            "status": "skipped",
            "message": "no LLM provider configured",
        }

    context = build_llm_diagnosis_context(
        failure=failure,
        diagnosis_summary=diagnosis_summary,
        possible_issue=possible_issue,
        retrieval_results=results,
    )
    prompt = build_llm_diagnosis_prompt(context)
    try:
        response = provider.diagnose(LLMRequest(prompt=prompt, model=model))
    except Exception:  # noqa: BLE001 - provider failures should not break diagnose JSON.
        return {
            "enabled": True,
            "provider": getattr(provider, "name", None),
            "model": model,
            "status": "error",
            "message": "LLM provider unavailable",
        }

    return {
        "enabled": True,
        "provider": response.provider,
        "model": response.model,
        "status": "completed",
        "text": response.text,
        "message": "completed",
    }


def _print_diagnose_status(status: str) -> None:
    print("CodeScope Diagnose")
    print()
    print("Status")
    print(f"- {status}")


def _print_failure_details(failure: TestFailure) -> None:
    print("Failing test")
    print(f"- [FAIL] {failure.test_name}")

    location = _failure_location(failure)
    if location:
        print(f"- File: {location}")

    print()
    print("Failure signal")
    if failure.error_type:
        print(f"- Error: {failure.error_type}")
    if failure.message:
        print(f"- Message: {failure.message}")
    print()


def _failure_location(failure: TestFailure) -> str:
    if not failure.file_path:
        return ""
    if failure.line_number is None:
        return failure.file_path
    return f"{failure.file_path}:{failure.line_number}"


def _print_retrieval_sections(
    results: list[RetrievalResult], *, repo_path: Path, failure: TestFailure | None = None
) -> None:
    semantic_results = [result for result in results if result.kind == "semantic"]
    related_results = [result for result in results if result.kind == "related"]

    print("Likely relevant code:")
    _print_retrieval_results(semantic_results, repo_path=repo_path, failure=failure)
    print()
    print("Related context:")
    _print_retrieval_results(related_results, repo_path=repo_path, failure=failure)


def _print_retrieval_results(
    results: list[RetrievalResult], *, repo_path: Path, failure: TestFailure | None
) -> None:
    if not results:
        print("- None")
        return

    for rank, result in enumerate(results, start=1):
        _print_retrieval_result(rank, result, repo_path=repo_path, failure=failure)


def _print_retrieval_result(
    rank: int, result: RetrievalResult, *, repo_path: Path, failure: TestFailure | None
) -> None:
    chunk = result.chunk
    chunk_name = _chunk_display_name(chunk)
    location = _chunk_location(chunk, repo_path=repo_path)

    print(f"{rank}. {chunk_name}")
    print(f"   Kind: {chunk.chunk_type}")
    print(f"   Location: {location}")
    print(f"   Source: {result.kind}")
    if result.score is not None:
        print(f"   Score: {result.score:.2f}")

    if failure is None:
        return

    reasons = build_retrieval_reasons(failure, chunk, extra_reasons=result.reasons)
    print("   reasons=")
    for reason in reasons:
        print(f"     - {reason}")


def _print_llm_diagnosis(
    *,
    failure: TestFailure,
    diagnosis_summary: str,
    possible_issue: str | None,
    results: list[RetrievalResult],
    provider: LLMProvider | None,
    model: str | None,
    provider_error: str | None,
) -> None:
    print("LLM Diagnosis")
    if provider_error:
        print("- Unavailable: LLM provider could not be loaded.")
        print(f"- Reason: {provider_error}")
        return

    if provider is None:
        print("- Skipped: no LLM provider configured.")
        print("- Set CODESCOPE_LLM_PROVIDER=fake to test this section.")
        return

    context = build_llm_diagnosis_context(
        failure=failure,
        diagnosis_summary=diagnosis_summary,
        possible_issue=possible_issue,
        retrieval_results=results,
    )
    prompt = build_llm_diagnosis_prompt(context)
    try:
        response = provider.diagnose(LLMRequest(prompt=prompt, model=model))
    except Exception as exc:  # noqa: BLE001 - provider failures should not break diagnose.
        print("- Unavailable: LLM diagnosis provider failed.")
        print(f"- Reason: {exc}")
        return

    print("AI-generated reasoning based only on retrieved CodeScope context.")
    print()
    print(response.text)


def _chunk_display_name(chunk: CodeChunk) -> str:
    if chunk.chunk_type == "method" and chunk.parent:
        return f"{chunk.parent}.{chunk.name}"
    return chunk.name


def _chunk_location(chunk: CodeChunk, *, repo_path: Path) -> str:
    return f"{_chunk_file_path(chunk, repo_path=repo_path)}:{chunk.start_line}-{chunk.end_line}"


def _chunk_file_path(chunk: CodeChunk, *, repo_path: Path) -> str:
    try:
        display_path = Path(chunk.file_path).relative_to(repo_path).as_posix()
    except ValueError:
        display_path = chunk.file_path
    return display_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "scan":
        return _handle_scan(args.repo_path)

    if args.command == "index":
        return _handle_index(args.repo_path)

    if args.command == "chunks":
        return _handle_chunks(args.repo_path)

    if args.command == "search":
        return _handle_search(args.repo_path, args.query, args.top_k)

    if args.command == "test":
        return _handle_test(args.repo_path, args.test_path)

    if args.command == "benchmark":
        return _handle_benchmark(args.benchmarks_path)

    if args.command == "diagnose":
        return _handle_diagnose(args.repo_path, use_llm=args.llm, json_output=args.json_output)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
