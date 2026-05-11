# CodeScope

CodeScope is a production-style, AI-powered codebase intelligence system.

This repository starts with a clean foundation and working core components:

- Repository scanner that discovers Python files while skipping common build/virtualenv directories
- AST-based structural chunk extraction (classes, functions, methods, and imports)

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

## Extract chunks

```bash
python -m codescope.cli chunks <repo_path>
```

## Run tests

```bash
python -m pytest
```

## Lint

```bash
ruff check .
```
