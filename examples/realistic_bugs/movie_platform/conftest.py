from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent


def pytest_configure() -> None:
    sys.path.insert(0, str(APP_ROOT))
    for module_name in list(sys.modules):
        if module_name == "app" or module_name.startswith("app."):
            del sys.modules[module_name]
