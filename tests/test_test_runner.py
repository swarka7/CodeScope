from __future__ import annotations

from pathlib import Path

import pytest

from codescope.cli import main as cli_main
from codescope.testing.failure_parser import FailureParser
from codescope.testing.test_runner import TestRunner


def test_passing_pytest_project(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    _write_pytest_project(
        repo_path,
        test_source="\n".join(
            [
                "def test_ok() -> None:",
                "    assert 1 + 1 == 2",
                "",
            ]
        ),
    )

    result = TestRunner().run(repo_path)

    assert result.exit_code == 0


def test_failing_pytest_project(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    _write_pytest_project(
        repo_path,
        test_source="\n".join(
            [
                "def test_fail() -> None:",
                "    assert 1 + 1 == 3",
                "",
            ]
        ),
    )

    result = TestRunner().run(repo_path)

    assert result.exit_code == 1
    assert "FAILED" in (result.stdout + result.stderr)


def test_failure_parser_extracts_test_name(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    test_source = "\n".join(
        [
            "def test_fail() -> None:",
            "    expected = 401",
            "    actual = 200",
            "    assert expected == actual",
            "",
        ]
    )
    _write_pytest_project(repo_path, test_source=test_source)

    run_result = TestRunner().run(repo_path)
    combined = (run_result.stdout + "\n" + run_result.stderr).strip()
    failures = FailureParser().parse(combined)

    assert run_result.exit_code == 1
    assert [f.test_name for f in failures] == ["tests/test_example.py::test_fail"]


def test_failure_parser_extracts_file_path_and_line_number(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    test_source = "\n".join(
        [
            "def test_fail() -> None:",
            "    expected = 401",
            "    actual = 200",
            "    assert expected == actual",
            "",
        ]
    )
    _write_pytest_project(repo_path, test_source=test_source)

    run_result = TestRunner().run(repo_path)
    combined = (run_result.stdout + "\n" + run_result.stderr).strip()
    failures = FailureParser().parse(combined)

    assert run_result.exit_code == 1
    assert len(failures) == 1
    assert failures[0].file_path == "tests/test_example.py"
    assert failures[0].line_number == 4
    assert failures[0].error_type == "AssertionError"
    assert "tests/test_example.py:4:" in failures[0].traceback.replace("\\", "/")


def test_graceful_behavior_when_no_failures_exist(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    _write_pytest_project(
        repo_path,
        test_source="\n".join(
            [
                "def test_ok() -> None:",
                "    assert True",
                "",
            ]
        ),
    )

    run_result = TestRunner().run(repo_path)
    combined = (run_result.stdout + "\n" + run_result.stderr).strip()
    failures = FailureParser().parse(combined)

    assert run_result.exit_code == 0
    assert failures == []


def test_cli_test_command_reports_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_path = tmp_path / "repo"
    _write_pytest_project(
        repo_path,
        test_source="\n".join(
            [
                "def test_fail() -> None:",
                "    assert False",
                "",
            ]
        ),
    )

    exit_code = cli_main(["test", str(repo_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Tests failed" in captured.out
    assert "[FAIL] tests/test_example.py::test_fail" in captured.out


def _write_pytest_project(repo_path: Path, *, test_source: str) -> None:
    repo_path.mkdir(parents=True, exist_ok=True)
    tests_dir = repo_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_example.py").write_text(test_source, encoding="utf-8")
