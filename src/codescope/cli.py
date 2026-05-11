from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from codescope.scanner.repo_scanner import RepoScanner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codescope")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan a repository for Python files")
    scan_parser.add_argument("repo_path", type=Path, help="Path to the repository root")

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


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "scan":
        return _handle_scan(args.repo_path)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
