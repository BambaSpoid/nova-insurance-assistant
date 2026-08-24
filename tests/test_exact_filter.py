from datetime import date

import pytest

from nova_assistant.domain import (
    DocumentCatalog,
    DocumentType,
    ProductType,
    load_catalog,
)
from nova_assistant.filtering import (
    SelectionRequest,
    SelectionStatus,
    select_corpus,
)


def test_selection_requires_product() -> None:
    result = select_corpus(SelectionRequest(version=2025))

    assert result.status is SelectionStatus.CLARIFICATION_REQUIRED
    assert result.documents == ()
    assert "produit" in result.reason.lower()


def test_selection_requires_version_or_contract_date() -> None:
    result = select_corpus(SelectionRequest(product=ProductType.HOME))

    assert result.status is SelectionStatus.CLARIFICATION_REQUIRED
    assert result.documents == ()
    assert "version" in result.reason.lower()
    assert "date" in result.reason.lower()


def test_selection_keeps_explicit_archived_version() -> None:
    result = select_corpus(
        SelectionRequest(
            product=ProductType.AUTO,
            version=2024,
        )
    )

    assert result.status is SelectionStatus.SELECTED
    assert result.allowed_document_ids == (
        "NOVA-AUTO-IPID-2024",
        "NOVA-AUTO-CG-2024",
    )
    assert all(document.status.value == "archived" for document in result.documents)


def test_selection_by_contract_date() -> None:
    result = select_corpus(
        SelectionRequest(
            product=ProductType.HOME,
            contract_date=date(2025, 6, 15),
        )
    )

    assert result.status is SelectionStatus.SELECTED
    assert result.allowed_document_ids == (
        "NOVA-HOME-IPID-2025",
        "NOVA-HOME-CG-2025",
    )


@pytest.mark.parametrize(
    "contract_date",
    [
        date(2025, 1, 1),
        date(2025, 12, 31),
    ],
)
def test_selection_includes_validity_boundaries(
    contract_date: date,
) -> None:
    result = select_corpus(
        SelectionRequest(
            product=ProductType.TRAVEL,
            contract_date=contract_date,
        )
    )

    assert result.status is SelectionStatus.SELECTED
    assert len(result.documents) == 3


def test_consistent_version_and_date_are_accepted() -> None:
    result = select_corpus(
        SelectionRequest(
            product=ProductType.AUTO,
            version=2025,
            contract_date=date(2025, 8, 1),
        )
    )

    assert result.status is SelectionStatus.SELECTED
    assert result.allowed_document_ids == (
        "NOVA-AUTO-IPID-2025",
        "NOVA-AUTO-CG-2025",
    )


def test_conflicting_version_and_date_are_rejected() -> None:
    result = select_corpus(
        SelectionRequest(
            product=ProductType.AUTO,
            version=2024,
            contract_date=date(2025, 8, 1),
        )
    )

    assert result.status is SelectionStatus.CONFLICTING_CONTEXT
    assert result.documents == ()


def test_selection_does_not_widen_unknown_version() -> None:
    result = select_corpus(
        SelectionRequest(
            product=ProductType.TRAVEL,
            version=2024,
        )
    )

    assert result.status is SelectionStatus.NO_MATCHING_CORPUS
    assert result.documents == ()


def test_selection_filters_document_types() -> None:
    result = select_corpus(
        SelectionRequest(
            product=ProductType.TRAVEL,
            version=2025,
            document_types=(DocumentType.FAQ,),
        )
    )

    assert result.status is SelectionStatus.SELECTED
    assert result.allowed_document_ids == ("NOVA-TRAVEL-FAQ-2025",)


def test_selection_excludes_non_searchable_documents() -> None:
    catalog = load_catalog()
    updated_documents = tuple(
        document.model_copy(update={"searchable": False})
        if (document.product is ProductType.AUTO and document.version == 2025)
        else document
        for document in catalog.documents
    )
    updated_catalog = DocumentCatalog(
        schema_version=catalog.schema_version,
        documents=updated_documents,
    )

    result = select_corpus(
        SelectionRequest(
            product=ProductType.AUTO,
            version=2025,
        ),
        catalog=updated_catalog,
    )

    assert result.status is SelectionStatus.NO_MATCHING_CORPUS
    assert result.documents == ()
