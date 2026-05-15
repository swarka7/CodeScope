from __future__ import annotations

from pathlib import Path

from codescope.graph.dependency_graph import DependencyGraph
from codescope.models.code_chunk import CodeChunk
from codescope.parser.ast_parser import AstParser
from codescope.parser.chunker import Chunker
from codescope.retrieval.dependency_aware import enrich_with_related
from codescope.vectorstore.memory_store import SearchResult


def test_extracts_function_call_dependencies(tmp_path: Path) -> None:
    source = "\n".join(
        [
            "def helper() -> None:",
            "    pass",
            "",
            "def main() -> None:",
            "    helper()",
            "",
        ]
    )
    file_path = tmp_path / "sample.py"
    file_path.write_text(source, encoding="utf-8")

    parsed = AstParser().parse_file(file_path)
    chunks = Chunker().extract_chunks(parsed)

    main_chunk = next(c for c in chunks if c.chunk_type == "function" and c.name == "main")
    assert "helper" in main_chunk.dependencies


def test_extracts_method_call_dependencies(tmp_path: Path) -> None:
    source = "\n".join(
        [
            "class Repo:",
            "    def save(self) -> None:",
            "        pass",
            "",
            "class Service:",
            "    def run(self) -> None:",
            "        self.repo.save()",
            "",
        ]
    )
    file_path = tmp_path / "sample.py"
    file_path.write_text(source, encoding="utf-8")

    parsed = AstParser().parse_file(file_path)
    chunks = Chunker().extract_chunks(parsed)

    run_chunk = next(
        c for c in chunks if c.chunk_type == "method" and c.parent == "Service" and c.name == "run"
    )
    assert "save" in run_chunk.dependencies
    assert ("repo.save" in run_chunk.dependencies) or ("self.repo.save" in run_chunk.dependencies)


def test_dependency_graph_exact_name_matching() -> None:
    helper = _chunk(name="helper", chunk_type="function", dependencies=[])
    main = _chunk(name="main", chunk_type="function", dependencies=["helper"])
    graph = DependencyGraph([helper, main])

    related = graph.related_chunks(main)
    assert [c.id for c in related] == [helper.id]


def test_dependency_graph_reverse_candidates_find_callers() -> None:
    validator = _chunk(
        name="check_status",
        chunk_type="function",
        dependencies=[],
        file_path="app/validators.py",
    )
    service = _chunk(
        name="update_status",
        chunk_type="function",
        dependencies=["v.check_status"],
        file_path="app/service.py",
        imports=["import app.validators as v"],
    )
    graph = DependencyGraph([validator, service])

    callers = graph.reverse_candidates(validator)

    assert [(dependency, chunk.id) for dependency, chunk in callers] == [
        ("check_status", service.id)
    ]


def test_search_enrichment_does_not_duplicate_results() -> None:
    helper = _chunk(name="helper", chunk_type="function", dependencies=[])
    main = _chunk(name="main", chunk_type="function", dependencies=["helper"])
    graph = DependencyGraph([helper, main])

    semantic_results = [
        SearchResult(chunk=main, score=0.9),
        SearchResult(chunk=helper, score=0.8),
    ]
    enriched = enrich_with_related(query="helper", semantic_results=semantic_results, graph=graph)

    assert [r.kind for r in enriched] == ["semantic", "semantic"]
    assert [r.chunk.id for r in enriched] == [main.id, helper.id]


def test_search_enrichment_adds_related_chunks() -> None:
    helper = _chunk(name="helper", chunk_type="function", dependencies=[])
    main = _chunk(name="main", chunk_type="function", dependencies=["helper"])
    graph = DependencyGraph([helper, main])

    semantic_results = [SearchResult(chunk=main, score=0.9)]
    enriched = enrich_with_related(query="helper", semantic_results=semantic_results, graph=graph)

    assert [r.kind for r in enriched] == ["semantic", "related"]
    assert [r.chunk.id for r in enriched] == [main.id, helper.id]


def test_search_enrichment_respects_max_related() -> None:
    helpers = [
        _chunk(name=f"h{i}", chunk_type="function", dependencies=[], file_path="app.py")
        for i in range(10)
    ]
    main = _chunk(
        name="main",
        chunk_type="function",
        dependencies=[h.name for h in helpers],
        file_path="app.py",
    )
    graph = DependencyGraph([main, *helpers])

    semantic_results = [SearchResult(chunk=main, score=0.9)]
    enriched = enrich_with_related(
        query="helpers",
        semantic_results=semantic_results,
        graph=graph,
        max_related=3,
    )

    related = [r for r in enriched if r.kind == "related"]
    assert len(related) == 3


def test_only_top_semantic_sources_expand_dependencies() -> None:
    dep_a = _chunk(name="dep_a", chunk_type="function", dependencies=[], file_path="a.py")
    dep_b = _chunk(name="dep_b", chunk_type="function", dependencies=[], file_path="b.py")

    a = _chunk(name="a", chunk_type="function", dependencies=["dep_a"], file_path="a.py")
    b = _chunk(name="b", chunk_type="function", dependencies=["dep_b"], file_path="b.py")

    graph = DependencyGraph([a, b, dep_a, dep_b])
    semantic_results = [
        SearchResult(chunk=a, score=0.9),
        SearchResult(chunk=b, score=0.8),
    ]

    enriched = enrich_with_related(
        query="deps",
        semantic_results=semantic_results,
        graph=graph,
        max_semantic_sources=1,
        max_related=10,
    )

    related_ids = [r.chunk.id for r in enriched if r.kind == "related"]
    assert related_ids == [dep_a.id]


def test_same_file_related_chunks_are_preferred() -> None:
    local = _chunk(name="local", chunk_type="function", dependencies=[], file_path="module.py")
    remote = _chunk(name="remote", chunk_type="function", dependencies=[], file_path="other.py")
    main = _chunk(
        name="main",
        chunk_type="function",
        dependencies=["remote", "local"],
        file_path="module.py",
    )

    graph = DependencyGraph([main, local, remote])
    semantic_results = [SearchResult(chunk=main, score=0.9)]

    enriched = enrich_with_related(
        query="main",
        semantic_results=semantic_results,
        graph=graph,
        max_related=2,
    )

    related_ids = [r.chunk.id for r in enriched if r.kind == "related"]
    assert related_ids == [local.id, remote.id]


def test_test_chunks_skipped_unless_query_mentions_tests() -> None:
    main = _chunk(
        name="main",
        chunk_type="function",
        dependencies=["test_helper"],
        file_path="app.py",
    )
    test_helper = _chunk(
        name="test_helper",
        chunk_type="function",
        dependencies=[],
        file_path="tests/test_app.py",
    )
    graph = DependencyGraph([main, test_helper])
    semantic_results = [SearchResult(chunk=main, score=0.9)]

    enriched = enrich_with_related(
        query="repository scanner",
        semantic_results=semantic_results,
        graph=graph,
    )
    assert [r.chunk.id for r in enriched] == [main.id]

    enriched = enrich_with_related(
        query="pytest repository scanner", semantic_results=semantic_results, graph=graph
    )
    assert [r.chunk.id for r in enriched] == [main.id, test_helper.id]


def test_duplicates_are_avoided_across_semantic_sources() -> None:
    shared = _chunk(name="shared", chunk_type="function", dependencies=[], file_path="shared.py")
    a = _chunk(name="a", chunk_type="function", dependencies=["shared"], file_path="a.py")
    b = _chunk(name="b", chunk_type="function", dependencies=["shared"], file_path="b.py")

    graph = DependencyGraph([a, b, shared])
    semantic_results = [
        SearchResult(chunk=a, score=0.9),
        SearchResult(chunk=b, score=0.8),
    ]

    enriched = enrich_with_related(
        query="shared",
        semantic_results=semantic_results,
        graph=graph,
        max_semantic_sources=2,
        max_related=5,
    )

    related = [r for r in enriched if r.kind == "related"]
    assert [r.chunk.id for r in related] == [shared.id]


def test_weak_infrastructure_related_chunks_are_filtered_out() -> None:
    semantic = _chunk(
        name="scan_repo",
        chunk_type="function",
        dependencies=["IndexStore.exists"],
        file_path="scanner/repo_scanner.py",
    )
    infra = _chunk(
        name="exists",
        parent="IndexStore",
        chunk_type="method",
        dependencies=[],
        file_path="src/codescope/indexing/index_store.py",
    )

    graph = DependencyGraph([semantic, infra])
    semantic_results = [SearchResult(chunk=semantic, score=0.9)]

    enriched = enrich_with_related(
        query="repository scanner",
        semantic_results=semantic_results,
        graph=graph,
    )

    assert [r.kind for r in enriched] == ["semantic"]


def test_exact_dependency_matches_are_ranked_higher() -> None:
    semantic = _chunk(
        name="main",
        chunk_type="function",
        dependencies=["Repo.save", "helper"],
        file_path="module.py",
    )
    helper = _chunk(
        name="helper",
        chunk_type="function",
        dependencies=[],
        file_path="module.py",
    )
    save = _chunk(
        name="save",
        parent="Repo",
        chunk_type="method",
        dependencies=[],
        file_path="module.py",
    )

    graph = DependencyGraph([semantic, helper, save])
    semantic_results = [SearchResult(chunk=semantic, score=0.9)]

    enriched = enrich_with_related(
        query="main",
        semantic_results=semantic_results,
        graph=graph,
        max_related=2,
    )

    related_ids = [r.chunk.id for r in enriched if r.kind == "related"]
    assert related_ids == [helper.id, save.id]


def test_semantic_results_remain_first() -> None:
    helper = _chunk(name="helper", chunk_type="function", dependencies=[], file_path="app.py")
    main = _chunk(
        name="main",
        chunk_type="function",
        dependencies=["helper"],
        file_path="app.py",
    )
    other = _chunk(name="other", chunk_type="function", dependencies=[], file_path="other.py")

    graph = DependencyGraph([main, helper, other])
    semantic_results = [
        SearchResult(chunk=main, score=0.9),
        SearchResult(chunk=other, score=0.8),
    ]

    enriched = enrich_with_related(query="main", semantic_results=semantic_results, graph=graph)

    assert [r.kind for r in enriched[:2]] == ["semantic", "semantic"]


def test_multi_hop_traversal_adds_indirect_dependencies() -> None:
    leaf = _chunk(name="leaf", chunk_type="function", dependencies=[], file_path="module.py")
    middle = _chunk(
        name="middle",
        chunk_type="function",
        dependencies=["leaf"],
        file_path="module.py",
    )
    root = _chunk(
        name="root",
        chunk_type="function",
        dependencies=["middle"],
        file_path="module.py",
    )

    graph = DependencyGraph([root, middle, leaf])
    semantic_results = [SearchResult(chunk=root, score=0.9)]

    enriched = enrich_with_related(
        query="root",
        semantic_results=semantic_results,
        graph=graph,
        max_depth=2,
        max_related=5,
    )

    assert [r.kind for r in enriched] == ["semantic", "related", "related"]
    assert [r.chunk.id for r in enriched] == [root.id, middle.id, leaf.id]


def test_depth_limit_prevents_indirect_dependencies() -> None:
    leaf = _chunk(name="leaf", chunk_type="function", dependencies=[], file_path="module.py")
    middle = _chunk(
        name="middle",
        chunk_type="function",
        dependencies=["leaf"],
        file_path="module.py",
    )
    root = _chunk(
        name="root",
        chunk_type="function",
        dependencies=["middle"],
        file_path="module.py",
    )

    graph = DependencyGraph([root, middle, leaf])
    semantic_results = [SearchResult(chunk=root, score=0.9)]

    enriched = enrich_with_related(
        query="root",
        semantic_results=semantic_results,
        graph=graph,
        max_depth=1,
        max_related=5,
    )

    assert [r.chunk.id for r in enriched] == [root.id, middle.id]


def test_multi_hop_duplicate_chunks_are_avoided() -> None:
    shared = _chunk(name="shared", chunk_type="function", dependencies=[], file_path="module.py")
    left = _chunk(
        name="left",
        chunk_type="function",
        dependencies=["shared"],
        file_path="module.py",
    )
    right = _chunk(
        name="right",
        chunk_type="function",
        dependencies=["shared"],
        file_path="module.py",
    )
    root = _chunk(
        name="root",
        chunk_type="function",
        dependencies=["left", "right"],
        file_path="module.py",
    )

    graph = DependencyGraph([root, left, right, shared])
    semantic_results = [SearchResult(chunk=root, score=0.9)]

    enriched = enrich_with_related(
        query="root",
        semantic_results=semantic_results,
        graph=graph,
        max_depth=2,
        max_related=10,
    )

    ids = [r.chunk.id for r in enriched]
    assert len(ids) == len(set(ids))
    assert shared.id in ids


def test_multi_hop_same_module_indirect_chunks_rank_higher() -> None:
    indirect_same = _chunk(
        name="indirect_same",
        chunk_type="function",
        dependencies=[],
        file_path="pkg/models.py",
    )
    indirect_other = _chunk(
        name="indirect_other",
        chunk_type="function",
        dependencies=[],
        file_path="other/models.py",
    )
    dep_same_file = _chunk(
        name="dep_same_file",
        chunk_type="function",
        dependencies=["indirect_same"],
        file_path="pkg/service.py",
    )
    dep_other_module = _chunk(
        name="dep_other_module",
        chunk_type="function",
        dependencies=["indirect_other"],
        file_path="other/dep.py",
    )
    root = _chunk(
        name="root",
        chunk_type="function",
        dependencies=["dep_same_file", "dep_other_module"],
        file_path="pkg/service.py",
    )

    graph = DependencyGraph([root, dep_same_file, dep_other_module, indirect_same, indirect_other])
    semantic_results = [SearchResult(chunk=root, score=0.9)]

    enriched = enrich_with_related(
        query="root",
        semantic_results=semantic_results,
        graph=graph,
        max_depth=2,
        max_related=10,
    )

    indirect_ids = [r.chunk.id for r in enriched if r.chunk.name.startswith("indirect_")]
    assert indirect_ids == [indirect_same.id, indirect_other.id]


def test_multi_hop_same_file_indirect_chunks_are_preferred() -> None:
    indirect_same_file = _chunk(
        name="indirect_same_file",
        chunk_type="function",
        dependencies=[],
        file_path="module.py",
    )
    indirect_remote = _chunk(
        name="indirect_remote",
        chunk_type="function",
        dependencies=[],
        file_path="other.py",
    )
    dep_remote = _chunk(
        name="dep_remote",
        chunk_type="function",
        dependencies=["indirect_same_file"],
        file_path="other.py",
    )
    dep_local = _chunk(
        name="dep_local",
        chunk_type="function",
        dependencies=["indirect_remote"],
        file_path="module.py",
    )
    root = _chunk(
        name="root",
        chunk_type="function",
        dependencies=["dep_remote", "dep_local"],
        file_path="module.py",
    )

    graph = DependencyGraph([root, dep_remote, dep_local, indirect_same_file, indirect_remote])
    semantic_results = [SearchResult(chunk=root, score=0.9)]

    enriched = enrich_with_related(
        query="root",
        semantic_results=semantic_results,
        graph=graph,
        max_depth=2,
        max_related=10,
    )

    indirect_ids = [r.chunk.id for r in enriched if r.chunk.name.startswith("indirect_")]
    assert indirect_ids == [indirect_same_file.id, indirect_remote.id]


def test_multi_hop_test_noise_suppression_still_works() -> None:
    helper = _chunk(
        name="helper",
        chunk_type="function",
        dependencies=["test_util"],
        file_path="app.py",
    )
    root = _chunk(name="root", chunk_type="function", dependencies=["helper"], file_path="app.py")
    test_util = _chunk(
        name="test_util",
        chunk_type="function",
        dependencies=[],
        file_path="tests/test_util.py",
    )

    graph = DependencyGraph([root, helper, test_util])
    semantic_results = [SearchResult(chunk=root, score=0.9)]

    enriched = enrich_with_related(
        query="root",
        semantic_results=semantic_results,
        graph=graph,
        max_depth=2,
        max_related=10,
    )
    assert test_util.id not in [r.chunk.id for r in enriched]

    enriched = enrich_with_related(
        query="pytest root",
        semantic_results=semantic_results,
        graph=graph,
        max_depth=2,
        max_related=10,
    )
    assert test_util.id in [r.chunk.id for r in enriched]


def test_multi_hop_deterministic_ordering() -> None:
    shared = _chunk(name="shared", chunk_type="function", dependencies=[], file_path="module.py")
    a = _chunk(name="a", chunk_type="function", dependencies=["shared"], file_path="module.py")
    b = _chunk(name="b", chunk_type="function", dependencies=["shared"], file_path="module.py")
    root = _chunk(
        name="root",
        chunk_type="function",
        dependencies=["a", "b"],
        file_path="module.py",
    )

    graph = DependencyGraph([root, a, b, shared])
    semantic_results = [SearchResult(chunk=root, score=0.9)]

    first = enrich_with_related(
        query="root",
        semantic_results=semantic_results,
        graph=graph,
        max_depth=2,
        max_related=10,
    )
    second = enrich_with_related(
        query="root",
        semantic_results=semantic_results,
        graph=graph,
        max_depth=2,
        max_related=10,
    )

    assert [r.chunk.id for r in first] == [r.chunk.id for r in second]


def _chunk(
    *,
    name: str,
    chunk_type: str,
    dependencies: list[str],
    file_path: str = "file.py",
    parent: str | None = None,
    imports: list[str] | None = None,
) -> CodeChunk:
    return CodeChunk(
        id=f"{chunk_type}:{name}",
        file_path=file_path,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        name=name,
        parent=parent,
        start_line=1,
        end_line=1,
        source_code="pass\n",
        imports=imports or [],
        dependencies=dependencies,
    )
