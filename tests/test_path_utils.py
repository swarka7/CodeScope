from __future__ import annotations

from codescope.utils.path_utils import display_path, is_test_path, normalize_path


def test_normalize_path_uses_forward_slashes_and_lowercase() -> None:
    assert normalize_path(r".\Src\Auth\Service.py") == "src/auth/service.py"


def test_is_test_path_detects_tests_directory() -> None:
    assert is_test_path("project/tests/test_auth.py") is True
    assert is_test_path(r"project\tests\helpers.py") is True


def test_is_test_path_detects_test_file_names() -> None:
    assert is_test_path("test_auth.py") is True
    assert is_test_path("auth_test.py") is True
    assert is_test_path("conftest.py") is True


def test_is_test_path_rejects_source_files() -> None:
    assert is_test_path("src/auth/service.py") is False
    assert is_test_path("contest.py") is False


def test_display_path_uses_forward_slashes() -> None:
    assert display_path(r"src\auth\service.py") == "src/auth/service.py"
