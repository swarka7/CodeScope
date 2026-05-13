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

## Semantic search

Install the embeddings extra (uses `sentence-transformers` locally):

```bash
python -m pip install -e ".[ai]"
```

Then search:

```bash
python -m codescope.cli search <repo_path> "your query"
```

## Diagnose demo (buggy calculator)

This repo includes a tiny intentionally broken example project at `examples/buggy_calculator/`.

Make sure you have embeddings installed:

```bash
python -m pip install -e ".[ai]"
```

Index it:

```bash
python -m codescope.cli index examples/buggy_calculator
```

Then run diagnostics:

```bash
python -m codescope.cli diagnose examples/buggy_calculator
```

You should see `Tests failed` and a likely relevant chunk for `calculate_discount` in `calculator.py`.

## Run tests

```bash
python -m pytest
```

## Lint

```bash
ruff check .
```
