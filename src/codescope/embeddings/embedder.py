from __future__ import annotations

import re
from typing import Any

from codescope.models.code_chunk import CodeChunk

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_DEPENDENCIES_IN_EMBEDDING_TEXT = 12


class Embedder:
    """Embeds code chunks using a local Sentence-Transformers model."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, model: Any | None = None) -> None:
        self._model_name = model_name
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def build_embedding_text(chunk: CodeChunk) -> str:
        qualified_name = f"{chunk.parent}.{chunk.name}" if chunk.parent else chunk.name
        header = [
            f"type: {chunk.chunk_type}",
            f"name: {qualified_name}",
            f"path: {chunk.file_path}:{chunk.start_line}-{chunk.end_line}",
        ]

        if chunk.imports:
            header.append("imports:")
            header.extend(f"- {imp}" for imp in chunk.imports)

        if chunk.decorators:
            header.append("decorators:")
            header.extend(f"- {decorator}" for decorator in chunk.decorators)
            route_hints = _fastapi_route_hints(chunk.decorators)
            if route_hints:
                header.append("framework hints:")
                header.extend(f"- {hint}" for hint in route_hints)

        dependencies = _limited_unique(chunk.dependencies, limit=MAX_DEPENDENCIES_IN_EMBEDDING_TEXT)
        if dependencies:
            header.append("dependencies:")
            header.extend(f"- {dependency}" for dependency in dependencies)

        header.append("source:")
        header.append(chunk.source_code.rstrip("\n"))

        return "\n".join(header).strip() + "\n"

    def embed_text(self, text: str) -> list[float]:
        model = self._get_or_load_model()
        embedding = self._encode_query(model=model, text=text)
        return _to_float_list(embedding)

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        if not chunks:
            return []

        model = self._get_or_load_model()
        texts = [self.build_embedding_text(chunk) for chunk in chunks]
        embeddings = self._encode_documents(model=model, texts=texts)

        return [_to_float_list(row) for row in embeddings]

    def _get_or_load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. Install it with: pip install -e \".[ai]\""
            ) from exc

        self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode_query(self, *, model: Any, text: str) -> Any:
        if hasattr(model, "encode_query"):
            return model.encode_query(text, normalize_embeddings=True)
        return model.encode(text, normalize_embeddings=True)

    def _encode_documents(self, *, model: Any, texts: list[str]) -> Any:
        if hasattr(model, "encode_document"):
            return model.encode_document(texts, normalize_embeddings=True)
        return model.encode(texts, normalize_embeddings=True)


def _to_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if isinstance(vector, list | tuple):
        return [float(x) for x in vector]
    raise TypeError(f"Unsupported embedding type: {type(vector)!r}")


def _fastapi_route_hints(decorators: list[str]) -> list[str]:
    hints: list[str] = []
    for decorator in decorators:
        match = re.match(
            r"^@\w+\.(get|post|put|patch|delete)\(\s*(['\"])([^'\"]*)\2",
            decorator,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        method = match.group(1).upper()
        route = match.group(3)
        hints.append(f"FastAPI route handler: {method} {route}")
        hints.append(f"{method} route endpoint")
    return hints


def _limited_unique(values: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_values.append(normalized)
        if len(unique_values) >= limit:
            break
    return unique_values
