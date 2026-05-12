from __future__ import annotations

import os
from pathlib import Path

import pytest


def _safe_name(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum() or ch in ("-", "_")).strip("_-") or "unknown"


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Force a writable per-user base temp directory for tmp_path.

    In some sandboxed Windows environments the default system temp directory can
    be inaccessible. We also avoid collisions across different OS users by
    including the login name in the basetemp path.
    """

    if getattr(config.option, "basetemp", None):
        return

    try:
        user = os.getlogin()
    except OSError:
        user = os.environ.get("USERNAME", "unknown")

    basetemp = Path(config.rootpath) / f".pytest_tmp_{_safe_name(user).lower()}"
    config.option.basetemp = str(basetemp)
