from __future__ import annotations

from pathlib import Path


def normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized.removeprefix("./")
    return normalized.strip("/").lower()


def is_test_path(path: str) -> bool:
    normalized = normalize_path(path)
    wrapped = f"/{normalized}/"
    if "/tests/" in wrapped:
        return True

    file_name = normalized.rsplit("/", 1)[-1]
    return file_name == "conftest.py" or file_name.startswith("test_") or file_name.endswith(
        "_test.py"
    )


def display_path(path: str) -> str:
    return Path(path).as_posix()
