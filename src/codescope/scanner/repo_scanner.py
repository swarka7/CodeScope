from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RepoScanner:
    """Discovers Python source files within a repository directory tree."""

    excluded_dir_names: frozenset[str] = frozenset(
        {
            ".git",
            ".venv",
            ".cache",
            ".pytest_tmp",
            "__pycache__",
            "node_modules",
            "dist",
            "build",
        }
    )

    def scan(self, repo_path: Path) -> list[Path]:
        repo_path = Path(repo_path)

        if not repo_path.exists():
            raise FileNotFoundError(f"Path does not exist: {repo_path}")
        if not repo_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {repo_path}")

        files = list(self._iter_python_files(repo_path))
        return sorted(files)

    def _iter_python_files(self, root: Path) -> Iterable[Path]:
        stack: list[Path] = [root]

        while stack:
            current_dir = stack.pop()
            try:
                for child in current_dir.iterdir():
                    if child.is_dir():
                        if child.name in self.excluded_dir_names:
                            continue
                        stack.append(child)
                        continue

                    if child.is_file() and child.suffix == ".py":
                        yield child
            except PermissionError:
                continue
