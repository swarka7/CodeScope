# CodeScope

CodeScope is a production-style, AI-powered codebase intelligence system.

This repository starts with a clean foundation and a first working component: a repository scanner that discovers
Python files while skipping common build/virtualenv directories.

## Setup

Create a virtual environment and install in editable mode:

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\python -m pip install -e ".[dev]"

# macOS/Linux
.venv/bin/python -m pip install -e ".[dev]"
```

Optional (future milestones):

```bash
# Add dependencies that will be used in later milestones
.\.venv\Scripts\python -m pip install -e ".[core,ai,vectorstore]"  # Windows
.venv/bin/python -m pip install -e ".[core,ai,vectorstore]"        # macOS/Linux
```

## Run the scanner

```bash
python -m codescope.cli scan <repo_path>
```

## Run tests

```bash
python -m pytest
```

## Lint

```bash
ruff check .
```
