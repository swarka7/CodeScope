from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from codescope.embeddings.embedder import Embedder
from codescope.graph.dependency_graph import DependencyGraph
from codescope.indexing.index_store import IndexStore
from codescope.parser.ast_parser import AstParser
from codescope.parser.chunker import Chunker
from codescope.retrieval.dependency_aware import enrich_with_related
from codescope.scanner.repo_scanner import RepoScanner
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
    scanner = RepoScanner()
    try:
        files = scanner.scan(repo_path)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    parser = AstParser()
    chunker = Chunker()

    chunks = []
    for file_path in files:
        parsed = parser.parse_file(file_path)
        chunks.extend(chunker.extract_chunks(parsed))

    if not chunks:
        print("No chunks found")
        return 0

    embedder = Embedder()
    try:
        embeddings = embedder.embed_chunks(chunks)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "files_indexed": len(files),
        "chunks_indexed": len(chunks),
    }

    store = IndexStore(repo_path)
    store.save(chunks=chunks, embeddings=embeddings, metadata=metadata)

    print(f"Indexed {len(chunks)} chunks from {len(files)} files")
    print("Saved index to .codescope/")
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
    if not index_store.exists():
        print(
            "No CodeScope index found. Run: python -m codescope.cli index <repo_path>",
            file=sys.stderr,
        )
        return 2

    try:
        chunks, embeddings, _metadata = index_store.load()
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    embedder = Embedder()
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

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
