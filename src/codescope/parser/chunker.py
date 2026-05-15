from __future__ import annotations

import ast
import hashlib

from codescope.models.code_chunk import CodeChunk
from codescope.parser.ast_parser import ParsedPythonFile


class Chunker:
    """Extracts structural chunks (classes/functions/methods) from a parsed file."""

    def extract_chunks(self, parsed_file: ParsedPythonFile) -> list[CodeChunk]:
        if parsed_file.module is None:
            return []

        file_path = parsed_file.file_path.as_posix()
        imports = _extract_imports(parsed_file.module)
        known_defs = _extract_known_definitions(parsed_file.module)
        source = parsed_file.source

        chunks: list[CodeChunk] = []

        for node in parsed_file.module.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                chunks.append(
                    _chunk_function(
                        node=node,
                        file_path=file_path,
                        source=source,
                        imports=imports,
                        known_defs=known_defs,
                    )
                )
            elif isinstance(node, ast.ClassDef):
                chunks.append(
                    _chunk_class(node=node, file_path=file_path, source=source, imports=imports)
                )
                chunks.extend(
                    _chunk_methods(
                        node=node,
                        file_path=file_path,
                        source=source,
                        imports=imports,
                        known_defs=known_defs,
                    )
                )

        return chunks


def _extract_imports(module: ast.Module) -> list[str]:
    imports: list[str] = []
    for node in module.body:
        if isinstance(node, ast.Import):
            names = ", ".join(_format_alias(alias) for alias in node.names)
            imports.append(f"import {names}")
        elif isinstance(node, ast.ImportFrom):
            from_part = _format_from_module(module=node.module, level=node.level)
            names = ", ".join(_format_alias(alias) for alias in node.names)
            imports.append(f"from {from_part} import {names}")
    return imports


def _format_alias(alias: ast.alias) -> str:
    if alias.asname:
        return f"{alias.name} as {alias.asname}"
    return alias.name


def _format_from_module(module: str | None, level: int) -> str:
    dots = "." * max(level, 0)
    if module:
        return f"{dots}{module}"
    return dots or module or ""


def _chunk_function(
    *,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: str,
    source: str,
    imports: list[str],
    known_defs: set[str],
) -> CodeChunk:
    start_line, end_line = _node_span(node)
    source_code = _slice_source(source=source, start_line=start_line, end_line=end_line)
    dependencies = _extract_dependencies(node=node, known_defs=known_defs)
    decorators = _extract_decorators(node=node, source=source)
    chunk_id = _make_chunk_id(
        file_path=file_path,
        chunk_type="function",
        name=node.name,
        parent=None,
        start_line=start_line,
        end_line=end_line,
    )
    return CodeChunk(
        id=chunk_id,
        file_path=file_path,
        chunk_type="function",
        name=node.name,
        parent=None,
        start_line=start_line,
        end_line=end_line,
        source_code=source_code,
        imports=list(imports),
        dependencies=dependencies,
        decorators=decorators,
    )


def _chunk_class(
    *, node: ast.ClassDef, file_path: str, source: str, imports: list[str]
) -> CodeChunk:
    start_line, end_line = _node_span(node)
    source_code = _slice_source(source=source, start_line=start_line, end_line=end_line)
    decorators = _extract_decorators(node=node, source=source)
    chunk_id = _make_chunk_id(
        file_path=file_path,
        chunk_type="class",
        name=node.name,
        parent=None,
        start_line=start_line,
        end_line=end_line,
    )
    return CodeChunk(
        id=chunk_id,
        file_path=file_path,
        chunk_type="class",
        name=node.name,
        parent=None,
        start_line=start_line,
        end_line=end_line,
        source_code=source_code,
        imports=list(imports),
        dependencies=[],
        decorators=decorators,
    )


def _chunk_methods(
    *,
    node: ast.ClassDef,
    file_path: str,
    source: str,
    imports: list[str],
    known_defs: set[str],
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for stmt in node.body:
        if not isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            continue

        start_line, end_line = _node_span(stmt)
        source_code = _slice_source(source=source, start_line=start_line, end_line=end_line)
        dependencies = _extract_dependencies(node=stmt, known_defs=known_defs)
        decorators = _extract_decorators(node=stmt, source=source)
        chunk_id = _make_chunk_id(
            file_path=file_path,
            chunk_type="method",
            name=stmt.name,
            parent=node.name,
            start_line=start_line,
            end_line=end_line,
        )
        chunks.append(
            CodeChunk(
                id=chunk_id,
                file_path=file_path,
                chunk_type="method",
                name=stmt.name,
                parent=node.name,
                start_line=start_line,
                end_line=end_line,
                source_code=source_code,
                imports=list(imports),
                dependencies=dependencies,
                decorators=decorators,
            )
        )
    return chunks


def _extract_decorators(
    *, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, source: str
) -> list[str]:
    decorators: list[str] = []
    for decorator in node.decorator_list:
        text = ast.get_source_segment(source, decorator)
        if text is None:
            text = ast.unparse(decorator)
        text = " ".join(text.strip().split())
        if text:
            decorators.append(f"@{text}")
    return decorators


def _extract_known_definitions(module: ast.Module) -> set[str]:
    known: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            known.add(node.name)

        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                    known.add(stmt.name)
    return known


def _extract_dependencies(
    *, node: ast.FunctionDef | ast.AsyncFunctionDef, known_defs: set[str]
) -> list[str]:
    extractor = _DependencyExtractor(known_defs=known_defs)
    extractor.visit(node)
    return extractor.dependencies


class _DependencyExtractor(ast.NodeVisitor):
    def __init__(self, *, known_defs: set[str]) -> None:
        self._known_defs = known_defs
        self._seen: set[str] = set()
        self.dependencies: list[str] = []

    def _add(self, value: str) -> None:
        if not value:
            return
        if value in self._seen:
            return
        self._seen.add(value)
        self.dependencies.append(value)

    def visit_Call(self, node: ast.Call) -> None:
        for dep in _call_target_variants(node.func):
            self._add(dep)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in self._known_defs:
            self._add(node.id)
        self.generic_visit(node)


def _call_target_variants(expr: ast.expr) -> list[str]:
    if isinstance(expr, ast.Name):
        return [expr.id]

    if isinstance(expr, ast.Attribute):
        chain = _format_attribute_chain(expr)
        if chain is None:
            return [expr.attr]

        variants = [chain, expr.attr]
        if chain.startswith("self."):
            variants.append(chain.removeprefix("self."))
        elif chain.startswith("cls."):
            variants.append(chain.removeprefix("cls."))
        return variants

    return []


def _format_attribute_chain(attr: ast.Attribute) -> str | None:
    parts: list[str] = [attr.attr]
    current: ast.expr = attr.value

    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value

    if isinstance(current, ast.Name):
        parts.append(current.id)
        parts.reverse()
        return ".".join(parts)

    return None


def _node_span(node: ast.AST) -> tuple[int, int]:
    start_line = getattr(node, "lineno", 1)

    decorator_list = getattr(node, "decorator_list", None)
    if decorator_list:
        decorator_lines = [getattr(dec, "lineno", start_line) for dec in decorator_list]
        start_line = min([start_line, *decorator_lines])

    end_line = getattr(node, "end_lineno", None)
    if isinstance(end_line, int) and end_line >= start_line:
        return start_line, end_line

    # Fallback for nodes missing end positions (uncommon in modern Python parsing).
    body = getattr(node, "body", None)
    if isinstance(body, list) and body:
        last = body[-1]
        last_end = getattr(last, "end_lineno", getattr(last, "lineno", start_line))
        if isinstance(last_end, int) and last_end >= start_line:
            return start_line, last_end

    return start_line, start_line


def _slice_source(*, source: str, start_line: int, end_line: int) -> str:
    lines = source.splitlines(keepends=True)
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    return "".join(lines[start_idx:end_idx])


def _make_chunk_id(
    *,
    file_path: str,
    chunk_type: str,
    name: str,
    parent: str | None,
    start_line: int,
    end_line: int,
) -> str:
    raw = f"{file_path}:{chunk_type}:{parent or ''}:{name}:{start_line}:{end_line}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()
