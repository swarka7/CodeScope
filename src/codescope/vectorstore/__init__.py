"""Vector store integrations.

Milestone 3 starts with an in-memory store for local development and tests.
"""

__all__ = ["MemoryStore", "SearchResult", "cosine_similarity"]

from .memory_store import MemoryStore, SearchResult, cosine_similarity
