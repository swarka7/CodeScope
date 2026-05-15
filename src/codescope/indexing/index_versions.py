from __future__ import annotations

from typing import Any

INDEX_SCHEMA_VERSION = 2
EMBEDDING_TEXT_VERSION = 2


def is_current_index_metadata(*, metadata: dict[str, Any], embedding_model_name: str) -> bool:
    return (
        metadata.get("index_schema_version") == INDEX_SCHEMA_VERSION
        and metadata.get("embedding_text_version") == EMBEDDING_TEXT_VERSION
        and metadata.get("embedding_model_name") == embedding_model_name
    )
