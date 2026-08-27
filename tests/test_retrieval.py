from datetime import date
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

import nova_assistant.retrieval.retriever as retriever_module
from nova_assistant.domain import ProductType, load_catalog
from nova_assistant.filtering import (
    SelectionRequest,
    select_corpus,
)
from nova_assistant.indexing import (
    IndexedChunk,
    SemanticSearchResult,
    VectorIndex,
)
from nova_assistant.ingestion import load_ingested_pages
from nova_assistant.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
    Retriever,
    load_or_build_default_retriever,
)


class FakeEmbedder:
    def __init__(
        self,
        query_embedding: NDArray[np.float32],
    ) -> None:
        self.query_embedding = query_embedding
        self.calls = 0

    def embed_query(
        self,
        query: str,
    ) -> NDArray[np.float32]:
        self.calls += 1
        return self.query_embedding


def make_chunk(document_id: str) -> IndexedChunk:
    page = next(page for page in load_ingested_pages() if page.document_id == document_id)
    words = page.text.split()[:20]

    return IndexedChunk.from_page(
        page=page,
        chunk_number=1,
        text=" ".join(words),
        word_start=0,
        word_end=len(words),
    )


def make_test_index() -> tuple[VectorIndex, tuple[IndexedChunk, ...]]:
    document_ids = (
        "NOVA-AUTO-IPID-2024",
        "NOVA-AUTO-CG-2024",
        "NOVA-AUTO-IPID-2025",
        "NOVA-AUTO-CG-2025",
        "NOVA-TRAVEL-IPID-2025",
    )
    chunks = tuple(make_chunk(document_id) for document_id in document_ids)
    embeddings = np.eye(len(chunks), dtype=np.float32)

    return VectorIndex(chunks, embeddings), chunks


def test_retrieval_respects_exact_selection() -> None:
    index, chunks = make_test_index()
    query_embedding = np.zeros(len(chunks), dtype=np.float32)
    query_embedding[2] = 1.0
    embedder = FakeEmbedder(query_embedding)

    retriever = Retriever(
        catalog=load_catalog(),
        index=index,
        embedder_factory=lambda: embedder,
    )
    result = retriever.retrieve(
        RetrievalRequest(
            question="Quelle est la franchise collision ?",
            selection_request=SelectionRequest(
                product=ProductType.AUTO,
                version=2025,
            ),
            top_k=2,
        )
    )

    assert result.status is RetrievalStatus.RETRIEVED
    assert len(result.matches) == 2
    assert result.matches[0].chunk.chunk_id == chunks[2].chunk_id
    assert all(match.chunk.version == 2025 for match in result.matches)
    assert embedder.calls == 1


def test_retrieval_does_not_embed_when_clarification_is_required() -> None:
    index, chunks = make_test_index()
    factory_calls = 0

    def embedder_factory() -> FakeEmbedder:
        nonlocal factory_calls
        factory_calls += 1
        return FakeEmbedder(np.ones(len(chunks), dtype=np.float32))

    retriever = Retriever(
        catalog=load_catalog(),
        index=index,
        embedder_factory=embedder_factory,
    )
    result = retriever.retrieve(
        RetrievalRequest(
            question="Quelle est la franchise ?",
            selection_request=SelectionRequest(
                product=ProductType.AUTO,
            ),
        )
    )

    assert result.status is RetrievalStatus.CLARIFICATION_REQUIRED
    assert result.matches == ()
    assert factory_calls == 0


def test_retrieval_does_not_embed_conflicting_context() -> None:
    index, chunks = make_test_index()
    factory_calls = 0

    def embedder_factory() -> FakeEmbedder:
        nonlocal factory_calls
        factory_calls += 1
        return FakeEmbedder(np.ones(len(chunks), dtype=np.float32))

    retriever = Retriever(
        catalog=load_catalog(),
        index=index,
        embedder_factory=embedder_factory,
    )
    result = retriever.retrieve(
        RetrievalRequest(
            question="Quelle est la franchise ?",
            selection_request=SelectionRequest(
                product=ProductType.AUTO,
                version=2024,
                contract_date=date(2025, 6, 1),
            ),
        )
    )

    assert result.status is RetrievalStatus.CONFLICTING_CONTEXT
    assert result.matches == ()
    assert factory_calls == 0


def test_retrieval_does_not_embed_unknown_corpus() -> None:
    index, chunks = make_test_index()
    factory_calls = 0

    def embedder_factory() -> FakeEmbedder:
        nonlocal factory_calls
        factory_calls += 1
        return FakeEmbedder(np.ones(len(chunks), dtype=np.float32))

    retriever = Retriever(
        catalog=load_catalog(),
        index=index,
        embedder_factory=embedder_factory,
    )
    result = retriever.retrieve(
        RetrievalRequest(
            question="Que couvre Travel 2024 ?",
            selection_request=SelectionRequest(
                product=ProductType.TRAVEL,
                version=2024,
            ),
        )
    )

    assert result.status is RetrievalStatus.NO_MATCHING_CORPUS
    assert result.matches == ()
    assert factory_calls == 0


def test_retriever_loads_embedder_only_once() -> None:
    index, chunks = make_test_index()
    embedder = FakeEmbedder(np.ones(len(chunks), dtype=np.float32))
    factory_calls = 0

    def embedder_factory() -> FakeEmbedder:
        nonlocal factory_calls
        factory_calls += 1
        return embedder

    retriever = Retriever(
        catalog=load_catalog(),
        index=index,
        embedder_factory=embedder_factory,
    )
    request = RetrievalRequest(
        question="Quelle est la franchise ?",
        selection_request=SelectionRequest(
            product=ProductType.AUTO,
            version=2025,
        ),
    )

    retriever.retrieve(request)
    retriever.retrieve(request)

    assert factory_calls == 1
    assert embedder.calls == 2


def test_retrieval_raises_when_index_has_no_allowed_passage() -> None:
    chunk = make_chunk("NOVA-TRAVEL-IPID-2025")
    index = VectorIndex(
        (chunk,),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    embedder = FakeEmbedder(np.asarray([1.0, 0.0], dtype=np.float32))
    retriever = Retriever(
        catalog=load_catalog(),
        index=index,
        embedder_factory=lambda: embedder,
    )

    with pytest.raises(RuntimeError, match="aucun passage"):
        retriever.retrieve(
            RetrievalRequest(
                question="Quelle est la franchise Auto ?",
                selection_request=SelectionRequest(
                    product=ProductType.AUTO,
                    version=2025,
                ),
            )
        )


def test_retrieval_result_rejects_unauthorized_passage() -> None:
    request = RetrievalRequest(
        question="Quelle est la franchise ?",
        selection_request=SelectionRequest(
            product=ProductType.AUTO,
            version=2025,
        ),
    )
    selection = select_corpus(request.selection_request)
    unauthorized_chunk = make_chunk("NOVA-AUTO-IPID-2024")
    unauthorized_match = SemanticSearchResult(
        chunk=unauthorized_chunk,
        score=0.9,
    )

    with pytest.raises(ValidationError, match="non autorisé"):
        RetrievalResult(
            request=request,
            status=RetrievalStatus.RETRIEVED,
            selection=selection,
            matches=(unauthorized_match,),
        )


def test_retrieval_request_rejects_blank_question() -> None:
    with pytest.raises(ValidationError, match="vide"):
        RetrievalRequest(
            question="   ",
            selection_request=SelectionRequest(
                product=ProductType.AUTO,
                version=2025,
            ),
        )


def test_load_or_build_retriever_uses_existing_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_retriever = object()
    load_retriever = Mock(return_value=expected_retriever)
    build_index = Mock()

    monkeypatch.setattr(
        retriever_module,
        "load_default_retriever",
        load_retriever,
    )
    monkeypatch.setattr(
        retriever_module,
        "build_semantic_index",
        build_index,
    )

    result = load_or_build_default_retriever()

    assert result is expected_retriever
    load_retriever.assert_called_once_with()
    build_index.assert_not_called()


def test_load_or_build_retriever_builds_missing_index_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_retriever = object()
    load_retriever = Mock(
        side_effect=(
            FileNotFoundError("Index absent."),
            expected_retriever,
        )
    )
    build_index = Mock()

    monkeypatch.setattr(
        retriever_module,
        "load_default_retriever",
        load_retriever,
    )
    monkeypatch.setattr(
        retriever_module,
        "build_semantic_index",
        build_index,
    )

    result = load_or_build_default_retriever()

    assert result is expected_retriever
    assert load_retriever.call_count == 2
    build_index.assert_called_once_with()


def test_load_or_build_retriever_does_not_hide_invalid_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_retriever = Mock(side_effect=ValueError("Empreinte invalide."))
    build_index = Mock()

    monkeypatch.setattr(
        retriever_module,
        "load_default_retriever",
        load_retriever,
    )
    monkeypatch.setattr(
        retriever_module,
        "build_semantic_index",
        build_index,
    )

    with pytest.raises(ValueError, match="Empreinte invalide"):
        load_or_build_default_retriever()

    load_retriever.assert_called_once_with()
    build_index.assert_not_called()
