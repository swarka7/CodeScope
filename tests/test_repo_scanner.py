from __future__ import annotations

from pathlib import Path

import pytest

from codescope.scanner.repo_scanner import RepoScanner


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# test file\n", encoding="utf-8")


def test_scans_normal_directories(tmp_path: Path) -> None:
    _touch(tmp_path / "a.py")
    _touch(tmp_path / "sub" / "b.py")
    _touch(tmp_path / "sub" / "c.txt")

    scanner = RepoScanner()
    files = scanner.scan(tmp_path)

    assert [p.relative_to(tmp_path).as_posix() for p in files] == ["a.py", "sub/b.py"]


def test_ignores_excluded_directories(tmp_path: Path) -> None:
    _touch(tmp_path / "keep.py")
    _touch(tmp_path / ".git" / "ignored.py")
    _touch(tmp_path / ".venv" / "ignored2.py")
    _touch(tmp_path / "__pycache__" / "ignored3.py")
    _touch(tmp_path / "node_modules" / "ignored4.py")
    _touch(tmp_path / "dist" / "ignored5.py")
    _touch(tmp_path / "build" / "ignored6.py")

    scanner = RepoScanner()
    files = scanner.scan(tmp_path)

    assert [p.relative_to(tmp_path).as_posix() for p in files] == ["keep.py"]


def test_invalid_path_handling(tmp_path: Path) -> None:
    scanner = RepoScanner()

    with pytest.raises(FileNotFoundError):
        scanner.scan(tmp_path / "does_not_exist")

    file_path = tmp_path / "file.py"
    _touch(file_path)

    with pytest.raises(NotADirectoryError):
        scanner.scan(file_path)

