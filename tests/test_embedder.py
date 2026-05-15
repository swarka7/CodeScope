from __future__ import annotations

from codescope.embeddings.embedder import MAX_DEPENDENCIES_IN_EMBEDDING_TEXT, Embedder
from codescope.models.code_chunk import CodeChunk


def test_embedding_text_formatting() -> None:
    chunk = CodeChunk(
        id="chunk-id",
        file_path="src/example.py",
        chunk_type="function",
        name="do_thing",
        parent=None,
        start_line=10,
        end_line=20,
        source_code="def do_thing():\n    return 1\n",
        imports=["import os", "from typing import Any"],
        dependencies=["validate_status_transition", "repository.get", "validate_status_transition"],
        decorators=['@app.post("/add")'],
    )

    text = Embedder.build_embedding_text(chunk)

    assert "type: function" in text
    assert "name: do_thing" in text
    assert "path: src/example.py:10-20" in text
    assert "imports:" in text
    assert "- import os" in text
    assert "- from typing import Any" in text
    assert "decorators:" in text
    assert '- @app.post("/add")' in text
    assert "FastAPI route handler: POST /add" in text
    assert "POST route endpoint" in text
    assert "dependencies:" in text
    assert "- validate_status_transition" in text
    assert "- repository.get" in text
    assert text.count("- validate_status_transition") == 1
    assert "source:" in text
    assert "def do_thing():" in text


def test_embedding_text_caps_dependencies_deterministically() -> None:
    dependencies = [
        f"dependency_{number}" for number in range(MAX_DEPENDENCIES_IN_EMBEDDING_TEXT + 3)
    ]
    chunk = CodeChunk(
        id="chunk-id",
        file_path="src/example.py",
        chunk_type="function",
        name="do_thing",
        parent=None,
        start_line=1,
        end_line=3,
        source_code="def do_thing():\n    return 1\n",
        imports=[],
        dependencies=dependencies,
    )

    text = Embedder.build_embedding_text(chunk)

    assert f"- dependency_{MAX_DEPENDENCIES_IN_EMBEDDING_TEXT - 1}" in text
    assert f"- dependency_{MAX_DEPENDENCIES_IN_EMBEDDING_TEXT}" not in text


def test_embedder_can_use_injected_model() -> None:
    class FakeModel:
        def encode_query(self, text: str, *, normalize_embeddings: bool) -> list[float]:
            assert normalize_embeddings is True
            return [1.0, 0.0, 0.0]

        def encode_document(
            self, texts: list[str], *, normalize_embeddings: bool
        ) -> list[list[float]]:
            assert normalize_embeddings is True
            return [[float(len(t)), 0.0, 0.0] for t in texts]

    chunk = CodeChunk(
        id="chunk-id",
        file_path="src/example.py",
        chunk_type="function",
        name="do_thing",
        parent=None,
        start_line=1,
        end_line=1,
        source_code="def do_thing():\n    return 1\n",
        imports=[],
        dependencies=[],
    )

    embedder = Embedder(model=FakeModel())
    chunk_vectors = embedder.embed_chunks([chunk])
    query_vector = embedder.embed_text("query")

    assert chunk_vectors == [[float(len(Embedder.build_embedding_text(chunk))), 0.0, 0.0]]
    assert query_vector == [1.0, 0.0, 0.0]
