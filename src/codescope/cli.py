from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from codescope.parser.ast_parser import AstParser
from codescope.parser.chunker import Chunker
from codescope.scanner.repo_scanner import RepoScanner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codescope")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan a repository for Python files")
    scan_parser.add_argument("repo_path", type=Path, help="Path to the repository root")

    chunks_parser = subparsers.add_parser(
        "chunks", help="Extract structural code chunks from a repository"
    )
    chunks_parser.add_argument("repo_path", type=Path, help="Path to the repository root")

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


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "scan":
        return _handle_scan(args.repo_path)

    if args.command == "chunks":
        return _handle_chunks(args.repo_path)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
