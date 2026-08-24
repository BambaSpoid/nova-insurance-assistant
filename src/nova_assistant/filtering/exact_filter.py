from nova_assistant.domain import (
    DocumentCatalog,
    DocumentMetadata,
    load_catalog,
)
from nova_assistant.filtering.models import (
    CorpusSelection,
    SelectionRequest,
    SelectionStatus,
)


def select_corpus(
    request: SelectionRequest,
    catalog: DocumentCatalog | None = None,
) -> CorpusSelection:
    """Sélectionne les documents applicables sans recherche sémantique."""

    selected_catalog = catalog or load_catalog()

    if request.product is None:
        return CorpusSelection(
            request=request,
            status=SelectionStatus.CLARIFICATION_REQUIRED,
            reason=("Le produit d’assurance est nécessaire pour sélectionner le corpus."),
        )

    if request.version is None and request.contract_date is None:
        return CorpusSelection(
            request=request,
            status=SelectionStatus.CLARIFICATION_REQUIRED,
            reason=("La version ou la date du contrat est nécessaire pour sélectionner le corpus."),
        )

    scoped_documents = tuple(
        document for document in selected_catalog.documents if _matches_scope(document, request)
    )

    if not scoped_documents:
        return CorpusSelection(
            request=request,
            status=SelectionStatus.NO_MATCHING_CORPUS,
            reason=("Aucun document ne correspond au produit, à la langue et aux types demandés."),
        )

    if request.version is not None and request.contract_date is not None:
        return _select_by_version_and_date(
            request=request,
            documents=scoped_documents,
        )

    if request.version is not None:
        matching_documents = tuple(
            document for document in scoped_documents if document.version == request.version
        )
    else:
        matching_documents = tuple(
            document
            for document in scoped_documents
            if (document.effective_from <= request.contract_date <= document.effective_to)
        )

    if not matching_documents:
        return CorpusSelection(
            request=request,
            status=SelectionStatus.NO_MATCHING_CORPUS,
            reason=("Aucun document ne correspond exactement à la version ou à la date demandée."),
        )

    return _successful_selection(request, matching_documents)


def _matches_scope(
    document: DocumentMetadata,
    request: SelectionRequest,
) -> bool:
    if document.product is not request.product:
        return False

    if document.language != request.language:
        return False

    if not document.searchable:
        return False

    if request.document_types is not None and document.document_type not in request.document_types:
        return False

    return True


def _select_by_version_and_date(
    request: SelectionRequest,
    documents: tuple[DocumentMetadata, ...],
) -> CorpusSelection:
    version_matches = tuple(
        document for document in documents if document.version == request.version
    )
    date_matches = tuple(
        document
        for document in documents
        if (document.effective_from <= request.contract_date <= document.effective_to)
    )
    exact_matches = tuple(document for document in version_matches if document in date_matches)

    if exact_matches:
        return _successful_selection(request, exact_matches)

    if version_matches or date_matches:
        return CorpusSelection(
            request=request,
            status=SelectionStatus.CONFLICTING_CONTEXT,
            reason=("La version et la date du contrat désignent des corpus différents."),
        )

    return CorpusSelection(
        request=request,
        status=SelectionStatus.NO_MATCHING_CORPUS,
        reason=("Aucun document ne correspond à la version et à la date demandées."),
    )


def _successful_selection(
    request: SelectionRequest,
    documents: tuple[DocumentMetadata, ...],
) -> CorpusSelection:
    return CorpusSelection(
        request=request,
        status=SelectionStatus.SELECTED,
        documents=documents,
        reason=(f"{len(documents)} document(s) correspondent exactement au contexte."),
    )
