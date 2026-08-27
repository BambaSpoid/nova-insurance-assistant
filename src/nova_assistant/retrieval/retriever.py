from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from nova_assistant.domain import (
    DocumentCatalog,
    load_catalog,
)
from nova_assistant.filtering import (
    SelectionStatus,
    select_corpus,
)
from nova_assistant.indexing import (
    SentenceTransformerEmbedder,
    VectorIndex,
    build_semantic_index,
    file_sha256,
    load_semantic_index,
)
from nova_assistant.ingestion import DEFAULT_OUTPUT_PATH
from nova_assistant.retrieval.models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
)


class QueryEmbedder(Protocol):
    def embed_query(
        self,
        query: str,
    ) -> NDArray[np.float32]: ...


class Retriever:
    """Applique le filtre exact avant toute recherche sémantique."""

    def __init__(
        self,
        catalog: DocumentCatalog,
        index: VectorIndex,
        embedder_factory: Callable[[], QueryEmbedder],
    ) -> None:
        self.catalog = catalog
        self.index = index
        self.embedder_factory = embedder_factory
        self._embedder: QueryEmbedder | None = None

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResult:
        selection = select_corpus(
            request=request.selection_request,
            catalog=self.catalog,
        )

        if selection.status is not SelectionStatus.SELECTED:
            return RetrievalResult(
                request=request,
                status=RetrievalStatus(selection.status.value),
                selection=selection,
            )

        query_embedding = self._get_embedder().embed_query(request.question)
        matches = self.index.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            allowed_document_ids=selection.allowed_document_ids,
        )

        if not matches:
            raise RuntimeError(
                "Le corpus est sélectionné, mais aucun passage correspondant n’existe dans l’index."
            )

        return RetrievalResult(
            request=request,
            status=RetrievalStatus.RETRIEVED,
            selection=selection,
            matches=matches,
        )

    def _get_embedder(self) -> QueryEmbedder:
        if self._embedder is None:
            self._embedder = self.embedder_factory()

        return self._embedder


def load_default_retriever() -> Retriever:
    """Charge le catalogue et l’index local vérifié."""

    catalog = load_catalog()
    index, manifest = load_semantic_index()

    current_pages_hash = file_sha256(DEFAULT_OUTPUT_PATH)

    if current_pages_hash != manifest.source_pages_sha256:
        raise ValueError("L’index sémantique ne correspond pas aux pages ingérées.")

    return Retriever(
        catalog=catalog,
        index=index,
        embedder_factory=lambda: SentenceTransformerEmbedder(model_name=manifest.model_name),
    )


def load_or_build_default_retriever() -> Retriever:
    """Construit l’index absent avant de charger le retriever."""

    try:
        return load_default_retriever()
    except FileNotFoundError:
        build_semantic_index()
        return load_default_retriever()
