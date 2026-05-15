from __future__ import annotations

from codescope.graph.dependency_graph import DependencyGraph
from codescope.graph.symbol_resolver import SymbolResolver
from codescope.models.code_chunk import CodeChunk


def test_from_import_alias_resolves_to_imported_symbol() -> None:
    token_decode = _chunk(
        id="token_manager:decode",
        file_path="token_manager.py",
        chunk_type="function",
        name="decode_token",
    )
    other_decode = _chunk(
        id="other:decode",
        file_path="other.py",
        chunk_type="function",
        name="decode_token",
    )

    source = _chunk(
        id="auth:validate",
        file_path="auth_service.py",
        chunk_type="function",
        name="validate_token",
        imports=["from token_manager import decode_token as dt"],
        dependencies=["dt"],
    )

    graph = DependencyGraph([source, token_decode, other_decode])
    related = graph.related_chunks(source)

    assert [chunk.id for chunk in related] == [token_decode.id]


def test_from_import_resolves_simple_name_when_symbol_is_ambiguous() -> None:
    token_decode = _chunk(
        id="token_manager:decode",
        file_path="token_manager.py",
        chunk_type="function",
        name="decode_token",
    )
    other_decode = _chunk(
        id="other:decode",
        file_path="other.py",
        chunk_type="function",
        name="decode_token",
    )

    source = _chunk(
        id="auth:validate",
        file_path="auth_service.py",
        chunk_type="function",
        name="validate_token",
        imports=["from token_manager import decode_token"],
        dependencies=["decode_token"],
    )

    graph = DependencyGraph([source, token_decode, other_decode])
    related = graph.related_chunks(source)

    assert [chunk.id for chunk in related] == [token_decode.id]


def test_import_module_alias_resolves_attribute_calls_to_module_file() -> None:
    token_decode = _chunk(
        id="token_manager:decode",
        file_path="token_manager.py",
        chunk_type="function",
        name="decode_token",
    )
    other_decode = _chunk(
        id="other:decode",
        file_path="other.py",
        chunk_type="function",
        name="decode_token",
    )

    source = _chunk(
        id="auth:validate",
        file_path="auth_service.py",
        chunk_type="function",
        name="validate_token",
        imports=["import token_manager as tm"],
        dependencies=["tm.decode_token"],
    )

    graph = DependencyGraph([source, token_decode, other_decode])
    related = graph.related_chunks(source)

    assert [chunk.id for chunk in related] == [token_decode.id]


def test_from_package_import_module_resolves_attribute_calls() -> None:
    app_validator = _chunk(
        id="app:validator",
        file_path="app/validators.py",
        chunk_type="function",
        name="check_status",
    )
    admin_validator = _chunk(
        id="admin:validator",
        file_path="admin/validators.py",
        chunk_type="function",
        name="check_status",
    )
    source = _chunk(
        id="app:service",
        file_path="app/service.py",
        chunk_type="function",
        name="update_status",
        imports=["from app import validators"],
        dependencies=["validators.check_status", "check_status"],
    )

    graph = DependencyGraph([source, app_validator, admin_validator])
    related = graph.related_chunks(source)

    assert [chunk.id for chunk in related] == [app_validator.id]


def test_qualified_import_alias_resolves_specific_package_module() -> None:
    app_validator = _chunk(
        id="app:validator",
        file_path="app/validators.py",
        chunk_type="function",
        name="check_status",
    )
    admin_validator = _chunk(
        id="admin:validator",
        file_path="admin/validators.py",
        chunk_type="function",
        name="check_status",
    )
    source = _chunk(
        id="app:service",
        file_path="app/service.py",
        chunk_type="function",
        name="update_status",
        imports=["import app.validators as v"],
        dependencies=["v.check_status", "check_status"],
    )

    graph = DependencyGraph([source, app_validator, admin_validator])
    related = graph.related_chunks(source)

    assert [chunk.id for chunk in related] == [app_validator.id]


def test_from_import_function_alias_resolves_specific_package_symbol() -> None:
    app_validator = _chunk(
        id="app:validator",
        file_path="app/validators.py",
        chunk_type="function",
        name="check_status",
    )
    admin_validator = _chunk(
        id="admin:validator",
        file_path="admin/validators.py",
        chunk_type="function",
        name="check_status",
    )
    source = _chunk(
        id="app:service",
        file_path="app/service.py",
        chunk_type="function",
        name="update_status",
        imports=["from app.validators import check_status as check"],
        dependencies=["check"],
    )

    graph = DependencyGraph([source, app_validator, admin_validator])
    related = graph.related_chunks(source)

    assert [chunk.id for chunk in related] == [app_validator.id]


def test_relative_import_resolves_sibling_module_symbol() -> None:
    app_validator = _chunk(
        id="app:validator",
        file_path="app/validators.py",
        chunk_type="function",
        name="check_status",
    )
    admin_validator = _chunk(
        id="admin:validator",
        file_path="admin/validators.py",
        chunk_type="function",
        name="check_status",
    )
    source = _chunk(
        id="app:service",
        file_path="app/service.py",
        chunk_type="function",
        name="update_status",
        imports=["from .validators import check_status"],
        dependencies=["check_status"],
    )

    graph = DependencyGraph([source, app_validator, admin_validator])
    related = graph.related_chunks(source)

    assert [chunk.id for chunk in related] == [app_validator.id]


def test_parent_relative_import_resolves_parent_package_symbol() -> None:
    validator = _chunk(
        id="core:validator",
        file_path="app/core/validators.py",
        chunk_type="function",
        name="check_status",
    )
    sibling_validator = _chunk(
        id="services:validator",
        file_path="app/services/validators.py",
        chunk_type="function",
        name="check_status",
    )
    source = _chunk(
        id="app:service",
        file_path="app/services/service.py",
        chunk_type="function",
        name="update_status",
        imports=["from ..core.validators import check_status"],
        dependencies=["check_status"],
    )

    graph = DependencyGraph([source, validator, sibling_validator])
    related = graph.related_chunks(source)

    assert [chunk.id for chunk in related] == [validator.id]


def test_ambiguous_duplicate_module_stems_do_not_fallback_to_all_matches() -> None:
    app_validator = _chunk(
        id="app:validator",
        file_path="app/validators.py",
        chunk_type="function",
        name="check_status",
    )
    admin_validator = _chunk(
        id="admin:validator",
        file_path="admin/validators.py",
        chunk_type="function",
        name="check_status",
    )
    source = _chunk(
        id="service",
        file_path="service.py",
        chunk_type="function",
        name="update_status",
        dependencies=["check_status"],
    )

    graph = DependencyGraph([source, app_validator, admin_validator])

    assert graph.related_chunks(source) == []


def test_class_method_exact_resolution_works_with_qualified_names() -> None:
    save = _chunk(
        id="repo:save",
        file_path="repo.py",
        chunk_type="method",
        name="save",
        parent="Repo",
    )
    run = _chunk(
        id="repo:run",
        file_path="repo.py",
        chunk_type="function",
        name="run",
        dependencies=["Repo.save"],
    )

    graph = DependencyGraph([run, save])
    related = graph.related_chunks(run)

    assert [chunk.id for chunk in related] == [save.id]


def test_same_file_symbol_is_preferred_over_other_files() -> None:
    helper_local = _chunk(
        id="a:helper",
        file_path="a.py",
        chunk_type="function",
        name="helper",
    )
    helper_other = _chunk(
        id="b:helper",
        file_path="b.py",
        chunk_type="function",
        name="helper",
    )
    source = _chunk(
        id="a:main",
        file_path="a.py",
        chunk_type="function",
        name="main",
        dependencies=["helper"],
    )

    graph = DependencyGraph([source, helper_local, helper_other])
    related = graph.related_chunks(source)

    assert [chunk.id for chunk in related] == [helper_local.id]


def test_ambiguous_short_method_names_do_not_resolve_to_single_candidate() -> None:
    save_repo = _chunk(
        id="repo:save",
        file_path="repo.py",
        chunk_type="method",
        name="save",
        parent="Repo",
    )
    save_audit = _chunk(
        id="audit:save",
        file_path="audit.py",
        chunk_type="method",
        name="save",
        parent="Audit",
    )
    source = _chunk(
        id="service:run",
        file_path="service.py",
        chunk_type="function",
        name="run",
        dependencies=["save"],
    )

    resolver = SymbolResolver([source, save_repo, save_audit])
    resolved = resolver.resolve("save", source_chunk=source)

    assert resolved == []


def test_cls_method_resolution_uses_same_class() -> None:
    helper = _chunk(
        id="service:helper",
        file_path="service.py",
        chunk_type="method",
        parent="TaskService",
        name="_coerce_status",
    )
    source = _chunk(
        id="service:from_payload",
        file_path="service.py",
        chunk_type="method",
        parent="TaskService",
        name="from_payload",
        dependencies=["cls._coerce_status", "_coerce_status"],
    )

    graph = DependencyGraph([source, helper])
    related = graph.related_chunks(source)

    assert [chunk.id for chunk in related] == [helper.id]


def _chunk(
    *,
    id: str,
    file_path: str,
    chunk_type: str,
    name: str,
    parent: str | None = None,
    imports: list[str] | None = None,
    dependencies: list[str] | None = None,
) -> CodeChunk:
    return CodeChunk(
        id=id,
        file_path=file_path,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        name=name,
        parent=parent,
        start_line=1,
        end_line=1,
        source_code="pass\n",
        imports=list(imports or []),
        dependencies=list(dependencies or []),
    )
