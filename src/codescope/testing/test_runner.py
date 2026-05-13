from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TestRunResult:
    stdout: str
    stderr: str
    exit_code: int


class TestRunner:
    """Runs pytest in a target repository and captures output.

    This runner is intentionally small and defensive: test failures are expected
    and should not crash CodeScope.
    """

    def run(self, repo_path: Path, test_path: Path | None = None) -> TestRunResult:
        repo_path = Path(repo_path)

        cmd: list[str] = [
            sys.executable,
            "-m",
            "pytest",
            "--no-header",
            "--rootdir",
            str(repo_path),
            "--confcutdir",
            str(repo_path),
        ]
        if test_path is not None:
            cmd.append(str(test_path))
        else:
            cmd.append(".")

        env = dict(os.environ)
        env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(repo_path),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as exc:
            return TestRunResult(stdout="", stderr=f"Failed to run pytest: {exc}", exit_code=3)

        return TestRunResult(
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            exit_code=int(completed.returncode),
        )
