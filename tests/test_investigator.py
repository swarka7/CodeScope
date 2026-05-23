from __future__ import annotations

from pathlib import Path

from codescope.indexing.index_store import IndexStore
from codescope.indexing.index_versions import EMBEDDING_TEXT_VERSION, INDEX_SCHEMA_VERSION
from codescope.investigation import (
    Investigator,
    build_investigation_query,
    score_investigation_result,
)
from codescope.models.code_chunk import CodeChunk
from codescope.vectorstore.memory_store import SearchResult


def test_build_query_preserves_user_description() -> None:
    query = build_investigation_query(" When transfers fail, the receiver balance stays low. ")

    assert "Bug description:" in query
    assert "When transfers fail, the receiver balance stays low." in query
    assert "Investigation intent:" in query
    assert "business logic" in query


def test_source_chunks_outrank_test_chunks(tmp_path: Path) -> None:
    source = _chunk(
        tmp_path,
        name="transfer_money",
        file_path="app/service.py",
        source="def transfer_money():\n    return update_balance()\n",
    )
    test = _chunk(
        tmp_path,
        name="test_transfer_money",
        file_path="tests/test_service.py",
        source="def test_transfer_money():\n    assert transfer_money()\n",
    )

    result = _build_index_and_investigate(
        tmp_path,
        [test, source],
        "transfer money balance update fails",
        top_k=2,
    )

    assert result.likely_relevant_code[0].name == "transfer_money"
    assert "test context" in result.likely_relevant_code[1].reasons


def test_business_operation_outranks_repository_plumbing(tmp_path: Path) -> None:
    service = _chunk(
        tmp_path,
        name="move_funds",
        parent="PaymentService",
        chunk_type="method",
        file_path="app/service.py",
        source=(
            "def move_funds(self, sender, receiver, amount):\n"
            "    sender.balance -= amount\n"
            "    receiver.balance += amount\n"
            "    self.repository.save(sender)\n"
        ),
        dependencies=["repository.save"],
    )
    repository = _chunk(
        tmp_path,
        name="get_account",
        parent="PaymentRepository",
        chunk_type="method",
        file_path="app/repository.py",
        source="def get_account(self, account_id):\n    return self.accounts[account_id]\n",
    )

    result = _build_index_and_investigate(
        tmp_path,
        [repository, service],
        "moving funds updates sender and receiver balance incorrectly",
        top_k=2,
    )

    assert result.likely_relevant_code[0].name == "PaymentService.move_funds"
    assert "business operation" in result.likely_relevant_code[0].reasons
    assert "data-access context" in result.likely_relevant_code[1].reasons


def test_transfer_description_surfaces_business_method_and_state_operation(
    tmp_path: Path,
) -> None:
    transfer = _chunk(
        tmp_path,
        name="transfer",
        parent="LedgerService",
        chunk_type="method",
        file_path="app/service.py",
        source=(
            "def transfer(self, sender, receiver, amount):\n"
            "    sender.debit(amount)\n"
            "    self.repository.save(sender)\n"
            "    self.repository.save(receiver)\n"
        ),
        dependencies=["sender.debit", "repository.save"],
    )
    credit = _chunk(
        tmp_path,
        name="credit",
        parent="Wallet",
        chunk_type="method",
        file_path="app/models.py",
        source="def credit(self, amount):\n    self.balance += amount\n",
    )
    debit = _chunk(
        tmp_path,
        name="debit",
        parent="Wallet",
        chunk_type="method",
        file_path="app/models.py",
        source="def debit(self, amount):\n    self.balance -= amount\n",
    )
    repository = _chunk(
        tmp_path,
        name="get_wallet",
        parent="WalletRepository",
        chunk_type="method",
        file_path="app/repository.py",
        source="def get_wallet(self, wallet_id):\n    return self.wallets[wallet_id]\n",
    )

    result = _build_index_and_investigate(
        tmp_path,
        [repository, debit, credit, transfer],
        "when transferring money, the receiver balance does not increase",
        top_k=3,
    )

    names = [item.name for item in result.likely_relevant_code]
    assert names[0] == "LedgerService.transfer"
    assert "Wallet.credit" in names
    assert "state update logic" in result.likely_relevant_code[0].reasons


def test_search_filter_description_surfaces_filter_method(tmp_path: Path) -> None:
    route = _chunk(
        tmp_path,
        name="list_movies",
        file_path="app/routes.py",
        source="def list_movies(service, criteria):\n    return service.search(criteria)\n",
        dependencies=["service.search"],
    )
    repository = _chunk(
        tmp_path,
        name="list_movies",
        parent="MovieRepository",
        chunk_type="method",
        file_path="app/repository.py",
        source="def list_movies(self):\n    return tuple(self.movies)\n",
    )
    search = _chunk(
        tmp_path,
        name="search",
        parent="CatalogSearchService",
        chunk_type="method",
        file_path="app/search.py",
        source=(
            "def search(self, criteria):\n"
            "    results = self.repository.list_movies()\n"
            "    if criteria.genre:\n"
            "        results = [movie for movie in results if movie.genre == criteria.genre]\n"
            "    if criteria.rating:\n"
            "        results = [movie for movie in results if movie.rating >= criteria.rating]\n"
            "    return results\n"
        ),
        dependencies=["repository.list_movies"],
    )

    result = _build_index_and_investigate(
        tmp_path,
        [route, repository, search],
        "filter movies by genre and rating returns wrong genres",
        top_k=3,
    )

    assert result.likely_relevant_code[0].name == "CatalogSearchService.search"
    assert "filtering logic" in result.likely_relevant_code[0].reasons


def test_dependency_enrichment_appends_related_context_without_duplicates(
    tmp_path: Path,
) -> None:
    service = _chunk(
        tmp_path,
        name="approve_order",
        parent="OrderService",
        chunk_type="method",
        file_path="app/service.py",
        source=(
            "def approve_order(self, order):\n"
            "    validate_order(order)\n"
            "    order.status = 'approved'\n"
        ),
        dependencies=["validate_order"],
    )
    validator = _chunk(
        tmp_path,
        name="validate_order",
        file_path="app/validators.py",
        source="def validate_order(order):\n    if not order.lines:\n        raise ValueError()\n",
    )

    result = _build_index_and_investigate(
        tmp_path,
        [service, validator],
        "order approval should validate invalid orders",
        top_k=1,
    )

    assert [item.name for item in result.likely_relevant_code] == ["OrderService.approve_order"]
    assert [item.name for item in result.related_context] == ["validate_order"]
    all_ids = [
        item.chunk.id
        for item in (*result.likely_relevant_code, *result.related_context)
    ]
    assert len(all_ids) == len(set(all_ids))


def test_investigation_ordering_is_deterministic_on_ties(tmp_path: Path) -> None:
    later = _chunk(
        tmp_path,
        name="process",
        file_path="app/z_service.py",
        source="def process():\n    return amount\n",
    )
    earlier = _chunk(
        tmp_path,
        name="process",
        file_path="app/a_service.py",
        source="def process():\n    return amount\n",
    )

    result = _build_index_and_investigate(
        tmp_path,
        [later, earlier],
        "process amount",
        top_k=2,
    )

    assert [item.file_path for item in result.likely_relevant_code] == [
        "app/a_service.py",
        "app/z_service.py",
    ]


def test_top_k_limits_semantic_results(tmp_path: Path) -> None:
    chunks = [
        _chunk(tmp_path, name=f"operation_{index}", file_path=f"app/{index}.py")
        for index in range(3)
    ]

    result = _build_index_and_investigate(tmp_path, chunks, "operation", top_k=2)

    assert len(result.likely_relevant_code) == 2


def test_score_investigation_result_records_repository_penalty(tmp_path: Path) -> None:
    repository = _chunk(
        tmp_path,
        name="get_invoice",
        parent="InvoiceRepository",
        chunk_type="method",
        file_path="app/repository.py",
        source="def get_invoice(self, invoice_id):\n    return self.invoices[invoice_id]\n",
    )

    scored = score_investigation_result(
        description="invoice total is not updated",
        result=SearchResult(chunk=repository, score=1.0),
    )

    assert "data-access context" in scored.reasons


def _build_index_and_investigate(
    tmp_path: Path,
    chunks: list[CodeChunk],
    description: str,
    *,
    top_k: int,
):
    repo_path = tmp_path / "repo"
    repo_path.mkdir(exist_ok=True)
    IndexStore(repo_path).save(
        chunks=chunks,
        embeddings=[[1.0] for _ in chunks],
        metadata={
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
            "embedding_model_name": "fake-investigator",
        },
    )
    return Investigator(repo_path, embedder=_FakeEmbedder()).investigate(description, top_k=top_k)


def _chunk(
    tmp_path: Path,
    *,
    name: str,
    file_path: str,
    source: str | None = None,
    parent: str | None = None,
    chunk_type: str = "function",
    dependencies: list[str] | None = None,
) -> CodeChunk:
    repo_path = tmp_path / "repo"
    full_path = repo_path / file_path
    return CodeChunk(
        id=f"{file_path}:{parent or ''}:{name}",
        file_path=str(full_path),
        chunk_type=chunk_type,  # type: ignore[arg-type]
        name=name,
        parent=parent,
        start_line=1,
        end_line=5,
        source_code=source or f"def {name}():\n    return None\n",
        imports=[],
        dependencies=dependencies or [],
    )


class _FakeEmbedder:
    @property
    def model_name(self) -> str:
        return "fake-investigator"

    def embed_text(self, text: str) -> list[float]:
        _ = text
        return [1.0]
