from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from codescope.debugging.diagnosis_summary import build_diagnosis_summary
from codescope.debugging.failure_retriever import FailureRetriever
from codescope.embeddings.embedder import Embedder
from codescope.graph.dependency_graph import DependencyGraph
from codescope.indexing.index_compatibility import check_index_compatibility
from codescope.indexing.index_store import IndexStore
from codescope.indexing.indexer import Indexer
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


def _handle_diagnose(repo_path: Path) -> int:
    runner = TestRunner()
    run_result = runner.run(repo_path)

    combined_output = "\n".join([run_result.stdout, run_result.stderr]).strip()

    if run_result.exit_code == 0:
        print("Tests passed")
        return 0

    print("Tests failed")
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

    for failure in failures:
        print(f"[FAIL] {failure.test_name}")
        if failure.error_type:
            print(f"Error: {failure.error_type}")
        if failure.message:
            print(f"Message: {failure.message}")
        print()

        if not compatibility.compatible:
            print(compatibility.message, file=sys.stderr)
            return 2

        try:
            results = retriever.retrieve(failure, top_k=5)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(build_diagnosis_summary(failure, results))
        print()
        print("Likely relevant code:")
        _print_retrieval_results(results, repo_path=repo_path)
        print()

    return run_result.exit_code


def _print_retrieval_results(results: list[RetrievalResult], *, repo_path: Path) -> None:
    for result in results:
        chunk = result.chunk
        kind = result.kind
        score = result.score

        if chunk.chunk_type == "method" and chunk.parent:
            chunk_name = f"{chunk.parent}.{chunk.name}"
        else:
            chunk_name = chunk.name

        try:
            display_path = Path(chunk.file_path).relative_to(repo_path).as_posix()
        except ValueError:
            display_path = chunk.file_path

        location = f"{display_path}:{chunk.start_line}-{chunk.end_line}"
        kind_tag = f"[{kind}]".ljust(10)
        type_tag = f"[{chunk.chunk_type}]"

        if kind == "semantic" and score is not None:
            print(f"{kind_tag} {type_tag} {chunk_name} {location} score={score:.2f}")
        else:
            print(f"{kind_tag} {type_tag} {chunk_name} {location}")


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

    if args.command == "diagnose":
        return _handle_diagnose(args.repo_path)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
