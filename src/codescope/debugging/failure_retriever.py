from __future__ import annotations

import re
from pathlib import Path

from codescope.embeddings.embedder import Embedder
from codescope.graph.dependency_graph import DependencyGraph
from codescope.indexing.index_store import IndexStore
from codescope.models.test_failure import TestFailure
from codescope.retrieval.dependency_aware import RetrievalResult, enrich_with_related
from codescope.vectorstore.memory_store import MemoryStore


class FailureRetriever:
    """Retrieves likely relevant code chunks for a given test failure."""

    _MAX_MESSAGE_CHARS = 400
    _MAX_TRACEBACK_EXCERPT_CHARS = 800
    _MAX_TRACEBACK_EXCERPT_LINES = 14
    _MAX_TRACEBACK_SYMBOLS = 12
    _MAX_SOURCE_HINTS = 10

    _FILE_LINE_RE = re.compile(
        r"""File\s+["'](?P<path>[^"']+?\.py)["'],\s+line\s+(?P<line>\d+)(?:,\s+in\s+(?P<func>[A-Za-z_]\w*))?"""
    )
    _PYTEST_LOCATION_RE = re.compile(r"""(?P<path>\S+?\.py):(?P<line>\d+):""")
    _IN_FUNCTION_RE = re.compile(r"""\bin\s+(?P<func>[A-Za-z_]\w*)\b""")
    _CALL_SYMBOL_RE = re.compile(r"""\b(?P<name>[A-Za-z_]\w*)\s*\(""")

    def __init__(
        self,
        repo_path: Path,
        *,
        index_store: IndexStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._repo_path = Path(repo_path)
        self._index_store = index_store or IndexStore(self._repo_path)
        self._embedder = embedder or Embedder()

    @staticmethod
    def build_failure_query(failure: TestFailure) -> str:
        """Build a high-signal, deterministic retrieval query from a test failure."""
        test_name = failure.test_name.strip()
        error_type = (failure.error_type or "").strip()
        message = failure.message.strip()
        traceback_text = failure.traceback.strip()

        if len(message) > FailureRetriever._MAX_MESSAGE_CHARS:
            message = message[: FailureRetriever._MAX_MESSAGE_CHARS].rstrip() + "..."

        symbols, source_hints = FailureRetriever._extract_traceback_hints(
            traceback_text,
            max_symbols=FailureRetriever._MAX_TRACEBACK_SYMBOLS,
            max_source_hints=FailureRetriever._MAX_SOURCE_HINTS,
        )

        if failure.file_path.strip():
            location = failure.file_path.strip()
            if failure.line_number is not None:
                location = f"{location}:{failure.line_number}"
            source_hints = FailureRetriever._merge_unique([location], source_hints)

        excerpt = FailureRetriever._build_traceback_excerpt(
            traceback_text,
            max_lines=FailureRetriever._MAX_TRACEBACK_EXCERPT_LINES,
            max_chars=FailureRetriever._MAX_TRACEBACK_EXCERPT_CHARS,
        )

        parts: list[str] = []
        parts.append("Test:")
        parts.append(test_name or "<unknown>")

        parts.append("")
        parts.append("Error:")
        parts.append(error_type or "<unknown>")

        parts.append("")
        parts.append("Message:")
        parts.append(message or "<none>")

        parts.append("")
        parts.append("Traceback symbols:")
        parts.extend(symbols if symbols else ["<none>"])

        parts.append("")
        parts.append("Source hints:")
        parts.extend(source_hints if source_hints else ["<none>"])

        if excerpt:
            parts.append("")
            parts.append("Traceback excerpt:")
            parts.append(excerpt)

        return "\n".join(parts).strip() + "\n"

    @staticmethod
    def build_query(failure: TestFailure) -> str:
        return FailureRetriever.build_failure_query(failure)

    @staticmethod
    def _merge_unique(left: list[str], right: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for raw in [*left, *right]:
            value = raw.strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(value)
        return merged

    @staticmethod
    def _build_traceback_excerpt(traceback_text: str, *, max_lines: int, max_chars: int) -> str:
        if not traceback_text:
            return ""

        lines = [line.rstrip() for line in traceback_text.splitlines() if line.strip()]
        if not lines:
            return ""

        filtered: list[str] = []
        for line in lines:
            stripped = line.strip()
            if FailureRetriever._FILE_LINE_RE.search(stripped):
                continue
            if FailureRetriever._PYTEST_LOCATION_RE.search(stripped):
                continue
            filtered.append(line)

        if not filtered:
            return ""

        tail = filtered[-max_lines:]
        excerpt = "\n".join(tail)
        if len(excerpt) <= max_chars:
            return excerpt

        truncated = excerpt[: max_chars].rstrip()
        return truncated + "..."

    @staticmethod
    def _extract_traceback_hints(
        traceback_text: str, *, max_symbols: int, max_source_hints: int
    ) -> tuple[list[str], list[str]]:
        if not traceback_text:
            return ([], [])

        noisy_symbols = {
            "assert",
            "repr",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "len",
            "range",
            "print",
        }

        symbols: list[str] = []
        source_hints: list[str] = []
        seen_symbols: set[str] = set()
        seen_hints: set[str] = set()

        def add_symbol(symbol: str) -> None:
            if len(symbols) >= max_symbols:
                return
            normalized = symbol.strip()
            if not normalized:
                return
            key = normalized.lower()
            if key in noisy_symbols:
                return
            if key in seen_symbols:
                return
            seen_symbols.add(key)
            symbols.append(normalized)

        def add_hint(path: str, line_number: int | None) -> None:
            if len(source_hints) >= max_source_hints:
                return
            normalized_path = path.strip().strip("\"'")
            if not normalized_path:
                return
            hint = normalized_path
            if line_number is not None:
                hint = f"{hint}:{line_number}"
            key = hint.replace("\\", "/").lower()
            if key in seen_hints:
                return
            seen_hints.add(key)
            source_hints.append(hint)

        for raw in traceback_text.splitlines():
            line = raw.strip()
            if not line:
                continue

            file_match = FailureRetriever._FILE_LINE_RE.search(line)
            if file_match:
                add_hint(file_match.group("path"), int(file_match.group("line")))
                func = file_match.group("func")
                if func:
                    add_symbol(func)

            loc_match = FailureRetriever._PYTEST_LOCATION_RE.search(line)
            if loc_match:
                add_hint(loc_match.group("path"), int(loc_match.group("line")))

            func_match = FailureRetriever._IN_FUNCTION_RE.search(line)
            if func_match:
                add_symbol(func_match.group("func"))

            for call in FailureRetriever._CALL_SYMBOL_RE.finditer(line):
                add_symbol(call.group("name"))

            if len(symbols) >= max_symbols and len(source_hints) >= max_source_hints:
                break

        return (symbols, source_hints)

    def retrieve(self, failure: TestFailure, *, top_k: int = 5) -> list[RetrievalResult]:
        chunks, embeddings, _metadata = self._index_store.load()

        query = self.build_failure_query(failure)
        query_embedding = self._embedder.embed_text(query)

        store = MemoryStore()
        store.add(chunks, embeddings)
        semantic = store.search(query_embedding, top_k=top_k)

        graph = DependencyGraph(chunks)
        return enrich_with_related(query=query, semantic_results=semantic, graph=graph)
