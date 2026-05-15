from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from codescope.indexing.index_store import IndexStore
from codescope.indexing.index_versions import EMBEDDING_TEXT_VERSION, INDEX_SCHEMA_VERSION

MISSING_INDEX_MESSAGE = (
    "No CodeScope index found. Run: python -m codescope.cli index <repo_path>"
)
OUTDATED_INDEX_MESSAGE = "Index is outdated. Run: python -m codescope.cli index <repo_path>"

IndexCompatibilityReason = Literal["compatible", "missing", "outdated"]


@dataclass(frozen=True, slots=True)
class IndexCompatibilityResult:
    compatible: bool
    reason: IndexCompatibilityReason
    message: str
    requires_rebuild: bool = False


def check_index_compatibility(
    *, index_store: IndexStore, embedding_model_name: str
) -> IndexCompatibilityResult:
    if not index_store.exists():
        return IndexCompatibilityResult(
            compatible=False,
            reason="missing",
            message=MISSING_INDEX_MESSAGE,
            requires_rebuild=False,
        )

    metadata = index_store.load_metadata()
    if (
        metadata.get("index_schema_version") != INDEX_SCHEMA_VERSION
        or metadata.get("embedding_text_version") != EMBEDDING_TEXT_VERSION
        or metadata.get("embedding_model_name") != embedding_model_name
    ):
        return IndexCompatibilityResult(
            compatible=False,
            reason="outdated",
            message=OUTDATED_INDEX_MESSAGE,
            requires_rebuild=True,
        )

    return IndexCompatibilityResult(compatible=True, reason="compatible", message="")
