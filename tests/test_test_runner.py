from __future__ import annotations

import sys
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


def test_test_runner_works_with_relative_repo_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    monkeypatch.chdir(tmp_path)
    result = TestRunner().run(Path("repo"))

    assert result.exit_code == 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path semantics")
def test_test_runner_works_with_windows_style_relative_repo_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_path = tmp_path / "workspace" / "repo"
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

    monkeypatch.chdir(tmp_path)
    result = TestRunner().run(Path("workspace\\repo"))

    assert result.exit_code == 0


def test_diagnose_works_for_buggy_calculator_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_path = tmp_path / "examples" / "buggy_calculator"
    repo_path.mkdir(parents=True, exist_ok=True)
    (repo_path / "calculator.py").write_text(
        "\n".join(
            [
                "def calculate_discount(price: int, percent: int) -> int:",
                "    return price - (price * percent)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    tests_dir = repo_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_calculator.py").write_text(
        "\n".join(
            [
                "from calculator import calculate_discount",
                "",
                "def test_discount() -> None:",
                "    assert calculate_discount(100, 10) == 90",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    exit_code = cli_main(["diagnose", "examples/buggy_calculator"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "CodeScope Diagnose" in captured.out
    assert "Tests failed" in captured.out
    assert "confcutdir must be a directory" not in (captured.out + captured.err)
    assert captured.err.strip() == (
        "No CodeScope index found. Run: python -m codescope.cli index <repo_path>"
    )


def test_failure_parser_message_extracts_assert_equals(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    _write_pytest_project(
        repo_path,
        test_source="\n".join(
            [
                "def test_fail() -> None:",
                "    assert 1 == 2",
                "",
            ]
        ),
    )

    run_result = TestRunner().run(repo_path)
    combined = (run_result.stdout + "\n" + run_result.stderr).strip()
    failures = FailureParser().parse(combined)

    assert run_result.exit_code == 1
    assert len(failures) == 1
    assert failures[0].message.startswith("assert")
    assert "1 == 2" in failures[0].message


def test_failure_parser_message_extracts_introspection_output(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir(parents=True, exist_ok=True)
    (repo_path / "calculator.py").write_text(
        "\n".join(
            [
                "def calculate_discount(price: int, percent: int) -> int:",
                "    return price - (price * percent)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    tests_dir = repo_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_example.py").write_text(
        "\n".join(
            [
                "from calculator import calculate_discount",
                "",
                "def test_discount() -> None:",
                "    assert calculate_discount(100, 10) == 90",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_result = TestRunner().run(repo_path)
    combined = (run_result.stdout + "\n" + run_result.stderr).strip()
    failures = FailureParser().parse(combined)

    assert run_result.exit_code == 1
    assert len(failures) == 1
    assert "-900" in failures[0].message
    assert "90" in failures[0].message


def test_failure_parser_message_handles_multiline_assertion_output(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    _write_pytest_project(
        repo_path,
        test_source="\n".join(
            [
                "def test_fail() -> None:",
                "    assert [1, 2, 3] == [1, 2, 4]",
                "",
            ]
        ),
    )

    run_result = TestRunner().run(repo_path)
    combined = (run_result.stdout + "\n" + run_result.stderr).strip()
    failures = FailureParser().parse(combined)

    assert run_result.exit_code == 1
    assert len(failures) == 1
    assert failures[0].message.startswith("assert")
    assert "==" in failures[0].message


def test_failure_parser_message_is_truncated_but_not_tiny(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    long_left = "a" * 600
    long_right = "b" * 600
    _write_pytest_project(
        repo_path,
        test_source="\n".join(
            [
                "def test_fail() -> None:",
                f"    assert {long_left!r} == {long_right!r}",
                "",
            ]
        ),
    )

    run_result = TestRunner().run(repo_path)
    combined = (run_result.stdout + "\n" + run_result.stderr).strip()
    failures = FailureParser().parse(combined)

    assert run_result.exit_code == 1
    assert len(failures) == 1
    assert len(failures[0].message) >= 20
    assert len(failures[0].message) <= 220


def test_failure_parser_non_assertion_error_messages_still_work(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    _write_pytest_project(
        repo_path,
        test_source="\n".join(
            [
                "def test_error() -> None:",
                "    raise ValueError('bad percent')",
                "",
            ]
        ),
    )

    run_result = TestRunner().run(repo_path)
    combined = (run_result.stdout + "\n" + run_result.stderr).strip()
    failures = FailureParser().parse(combined)

    assert run_result.exit_code == 1
    assert len(failures) == 1
    assert failures[0].error_type == "ValueError"
    assert "bad percent" in failures[0].message


def _write_pytest_project(repo_path: Path, *, test_source: str) -> None:
    repo_path.mkdir(parents=True, exist_ok=True)
    tests_dir = repo_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_example.py").write_text(test_source, encoding="utf-8")
