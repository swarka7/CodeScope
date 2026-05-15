from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from codescope.models.code_chunk import CodeChunk


@dataclass(frozen=True, slots=True)
class ImportContext:
    module_aliases: dict[str, str]
    imported_symbols: dict[str, str]


@dataclass(frozen=True, slots=True)
class ResolvedSymbol:
    matched_name: str
    chunk: CodeChunk


class SymbolResolver:
    """A small, static symbol resolver for dependency-aware retrieval.

    This is intentionally heuristic-based and limited:
    - exact name matching
    - same-file preference
    - basic handling of import aliases and from-imports
    - no runtime import execution or type inference
    """

    def __init__(self, chunks: list[CodeChunk]) -> None:
        self._chunks_by_name: dict[str, list[CodeChunk]] = {}
        self._chunks_by_file: dict[str, dict[str, list[CodeChunk]]] = {}
        self._unique_methods: dict[str, CodeChunk] = {}
        self._files_by_module_path: dict[str, set[str]] = {}

        self._build_tables(chunks)

    def resolve(self, dependency_name: str, *, source_chunk: CodeChunk) -> list[ResolvedSymbol]:
        dep = (dependency_name or "").strip()
        if not dep:
            return []

        file_key = _normalize_file_key(source_chunk.file_path)
        import_context = _parse_import_context(source_chunk.imports)

        dotted = _split_dotted(dep)
        if len(dotted) >= 2:
            head, tail = dotted[0], dotted[-1]
            if head in {"self", "cls"} and source_chunk.parent:
                resolved_self = self._resolve_self_method(
                    parent=source_chunk.parent,
                    tail=tail,
                    source_file_key=file_key,
                )
                if resolved_self:
                    return resolved_self

            resolved = self._resolve_dotted(
                head=head, tail=tail, source_file_key=file_key, import_context=import_context
            )
            if resolved:
                return resolved
            return self._resolve_simple(
                name=dep, source_file_key=file_key, import_context=import_context
            )

        return self._resolve_simple(
            name=dep, source_file_key=file_key, import_context=import_context
        )

    def _resolve_dotted(
        self, *, head: str, tail: str, source_file_key: str, import_context: ImportContext
    ) -> list[ResolvedSymbol]:
        module_full = import_context.module_aliases.get(
            head
        ) or import_context.imported_symbols.get(head)
        if not module_full:
            return []

        module_path = _resolve_import_module(module_full, source_file_key=source_file_key)
        if not module_path:
            return []

        module_files = self._choose_module_files(module_path, source_file_key=source_file_key)
        if not module_files:
            return []

        resolved: list[ResolvedSymbol] = []
        for file_key in module_files:
            candidates = self._chunks_by_file.get(file_key, {}).get(tail, [])
            resolved.extend(ResolvedSymbol(matched_name=tail, chunk=chunk) for chunk in candidates)

        return _dedupe_and_sort(resolved)

    def _resolve_simple(
        self, *, name: str, source_file_key: str, import_context: ImportContext
    ) -> list[ResolvedSymbol]:
        # Prefer file-local symbols (including Class.method keys).
        file_symbols = self._chunks_by_file.get(source_file_key, {})
        file_matches = file_symbols.get(name, [])
        if file_matches:
            return _dedupe_and_sort(
                [ResolvedSymbol(matched_name=name, chunk=chunk) for chunk in file_matches]
            )

        imported_target = import_context.imported_symbols.get(name)
        if imported_target:
            module_name, original = _split_import_target(imported_target)
            module_path = _resolve_import_module(module_name, source_file_key=source_file_key)
            module_files = self._choose_module_files(module_path, source_file_key=source_file_key)
            resolved: list[ResolvedSymbol] = []
            for file_key in module_files:
                candidates = self._chunks_by_file.get(file_key, {}).get(original, [])
                resolved.extend(
                    ResolvedSymbol(matched_name=original, chunk=chunk) for chunk in candidates
                )
            if resolved:
                return _dedupe_and_sort(resolved)

        imported_module_matches = self._resolve_imported_module_member(
            name=name,
            source_file_key=source_file_key,
            import_context=import_context,
        )
        if imported_module_matches:
            return imported_module_matches

        # As a last step, resolve unambiguous method names (short form).
        unique_method = self._unique_methods.get(name)
        if unique_method is not None:
            return [ResolvedSymbol(matched_name=name, chunk=unique_method)]

        return []

    def _resolve_self_method(
        self, *, parent: str, tail: str, source_file_key: str
    ) -> list[ResolvedSymbol]:
        qualified = f"{parent}.{tail}"
        candidates = self._chunks_by_file.get(source_file_key, {}).get(qualified, [])
        if not candidates:
            return []
        return _dedupe_and_sort(
            [ResolvedSymbol(matched_name=qualified, chunk=chunk) for chunk in candidates]
        )

    def _resolve_imported_module_member(
        self, *, name: str, source_file_key: str, import_context: ImportContext
    ) -> list[ResolvedSymbol]:
        module_names = list(import_context.module_aliases.values())
        module_names.extend(
            target
            for local_name, target in import_context.imported_symbols.items()
            if target.rsplit(".", 1)[-1] == local_name
        )

        resolved: list[ResolvedSymbol] = []
        for module_name in _dedupe_strings(module_names):
            module_path = _resolve_import_module(module_name, source_file_key=source_file_key)
            module_files = self._choose_module_files(module_path, source_file_key=source_file_key)
            for file_key in module_files:
                candidates = self._chunks_by_file.get(file_key, {}).get(name, [])
                resolved.extend(
                    ResolvedSymbol(matched_name=name, chunk=chunk) for chunk in candidates
                )

        unique = _dedupe_and_sort(resolved)
        if len({item.chunk.id for item in unique}) == 1:
            return unique
        return []

    def _choose_module_files(self, module_path: str, *, source_file_key: str) -> list[str]:
        normalized_module_path = _normalize_module_path(module_path)
        candidates = sorted(self._files_by_module_path.get(normalized_module_path, set()))
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates

        source_dir = Path(source_file_key).parent
        same_dir = [path for path in candidates if Path(path).parent == source_dir]
        if len(same_dir) == 1:
            return same_dir

        return []

    def _build_tables(self, chunks: list[CodeChunk]) -> None:
        method_counts: dict[str, int] = {}
        method_last: dict[str, CodeChunk] = {}

        for chunk in chunks:
            file_key = _normalize_file_key(chunk.file_path)
            for module_path in _module_paths_for_file(file_key):
                self._files_by_module_path.setdefault(module_path, set()).add(file_key)

            self._chunks_by_name.setdefault(chunk.name, []).append(chunk)
            self._chunks_by_file.setdefault(file_key, {}).setdefault(chunk.name, []).append(chunk)

            if chunk.parent:
                qualified = f"{chunk.parent}.{chunk.name}"
                self._chunks_by_name.setdefault(qualified, []).append(chunk)
                self._chunks_by_file.setdefault(file_key, {}).setdefault(qualified, []).append(
                    chunk
                )

            if chunk.chunk_type == "method":
                method_counts[chunk.name] = method_counts.get(chunk.name, 0) + 1
                method_last[chunk.name] = chunk

        for name, count in method_counts.items():
            if count == 1:
                self._unique_methods[name] = method_last[name]


def _parse_import_context(import_lines: list[str]) -> ImportContext:
    module_aliases: dict[str, str] = {}
    imported_symbols: dict[str, str] = {}

    for line in import_lines:
        line = (line or "").strip()
        if not line:
            continue

        try:
            parsed = ast.parse(line, mode="exec")
        except SyntaxError:
            continue

        if not parsed.body:
            continue

        stmt = parsed.body[0]
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                module = alias.name
                bound = alias.asname or _module_stem(module)
                module_aliases[bound] = module
                if bound == _module_stem(module):
                    module_aliases.setdefault(_module_stem(module), module)
        elif isinstance(stmt, ast.ImportFrom):
            if not stmt.module and stmt.level == 0:
                continue
            module = _format_import_from_module(stmt.module, stmt.level)
            for alias in stmt.names:
                local = alias.asname or alias.name
                imported_symbols[local] = f"{module}.{alias.name}"

    return ImportContext(module_aliases=module_aliases, imported_symbols=imported_symbols)


def _split_dotted(value: str) -> list[str]:
    parts = [part for part in value.split(".") if part]
    return parts


def _module_stem(module: str) -> str:
    return _normalize_module_path(module).rsplit(".", 1)[-1]


def _split_import_target(value: str) -> tuple[str, str]:
    leading_dots = len(value) - len(value.lstrip("."))
    prefix = "." * leading_dots
    parts = _split_dotted(value.lstrip("."))
    if len(parts) >= 2:
        return f"{prefix}{'.'.join(parts[:-1])}", parts[-1]
    if parts:
        return prefix, parts[-1]
    return "", value


def _normalize_file_key(file_path: str) -> str:
    return file_path.replace("\\", "/")


def _normalize_module_path(module: str) -> str:
    return ".".join(part for part in module.strip().strip(".").split(".") if part)


def _format_import_from_module(module: str | None, level: int) -> str:
    return f"{'.' * max(level, 0)}{module or ''}"


def _resolve_import_module(module: str, *, source_file_key: str) -> str:
    module = (module or "").strip()
    if not module:
        return ""
    if not module.startswith("."):
        return _normalize_module_path(module)

    level = len(module) - len(module.lstrip("."))
    relative_tail = _normalize_module_path(module[level:])
    package_parts = _source_package_parts(source_file_key)
    keep_count = max(len(package_parts) - max(level - 1, 0), 0)
    base_parts = package_parts[:keep_count]
    if relative_tail:
        base_parts.extend(relative_tail.split("."))
    return _normalize_module_path(".".join(base_parts))


def _source_package_parts(source_file_key: str) -> list[str]:
    path = Path(_normalize_file_key(source_file_key))
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    elif parts:
        parts = parts[:-1]
    return [_module_part(part) for part in parts if _module_part(part)]


def _module_paths_for_file(file_key: str) -> list[str]:
    path = Path(_normalize_file_key(file_key))
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]

    module_parts = [_module_part(part) for part in parts if _module_part(part)]
    module_paths: list[str] = []
    for index in range(len(module_parts)):
        module_paths.append(".".join(module_parts[index:]))
    return _dedupe_strings(module_paths)


def _module_part(value: str) -> str:
    return value.replace(":", "").strip()


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalize_module_path(value)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _dedupe_and_sort(items: list[ResolvedSymbol]) -> list[ResolvedSymbol]:
    seen: set[str] = set()
    unique: list[ResolvedSymbol] = []
    for item in items:
        if item.chunk.id in seen:
            continue
        seen.add(item.chunk.id)
        unique.append(item)

    unique.sort(
        key=lambda item: (
            item.chunk.file_path,
            item.chunk.chunk_type,
            item.chunk.parent or "",
            item.chunk.name,
        )
    )
    return unique
