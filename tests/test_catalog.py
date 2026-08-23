import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from nova_assistant.domain import (
    DocumentCatalog,
    DocumentStatus,
    ProductType,
    load_catalog,
)

CATALOG_PATH = Path("data/catalog/document_catalog.json")


@pytest.fixture
def catalog_data() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def test_catalog_contains_expected_documents() -> None:
    catalog = load_catalog()

    assert catalog.schema_version == "1.0"
    assert len(catalog.documents) == 11

    product_counts = Counter(document.product for document in catalog.documents)
    assert product_counts == {
        ProductType.HOME: 4,
        ProductType.AUTO: 4,
        ProductType.TRAVEL: 3,
    }


def test_catalog_versions_and_statuses_are_consistent() -> None:
    catalog = load_catalog()

    version_counts = Counter(document.version for document in catalog.documents)
    status_counts = Counter(document.status for document in catalog.documents)

    assert version_counts == {2024: 4, 2025: 7}
    assert status_counts == {
        DocumentStatus.ARCHIVED: 4,
        DocumentStatus.ACTIVE: 7,
    }


def test_catalog_referenced_files_exist() -> None:
    catalog = load_catalog()

    catalog.validate_files_exist()


def test_catalog_rejects_duplicate_document_id(catalog_data: dict) -> None:
    duplicated_document = catalog_data["documents"][0].copy()
    duplicated_document["source_path"] = "data/corpus/source/home/2024/nova_home_duplicate_2024.md"
    duplicated_document["pdf_path"] = "data/corpus/pdf/home/2024/nova_home_duplicate_2024.pdf"
    catalog_data["documents"].append(duplicated_document)

    with pytest.raises(ValidationError, match="document_id"):
        DocumentCatalog.model_validate(catalog_data)


def test_catalog_rejects_inconsistent_version_status(
    catalog_data: dict,
) -> None:
    catalog_data["documents"][0]["status"] = "active"

    with pytest.raises(ValidationError, match="archived"):
        DocumentCatalog.model_validate(catalog_data)


def test_catalog_rejects_inconsistent_validity_period(
    catalog_data: dict,
) -> None:
    catalog_data["documents"][0]["effective_from"] = "2025-01-01"
    catalog_data["documents"][0]["effective_to"] = "2025-12-31"

    with pytest.raises(ValidationError, match="version"):
        DocumentCatalog.model_validate(catalog_data)


def test_catalog_rejects_absolute_paths(catalog_data: dict) -> None:
    catalog_data["documents"][0]["source_path"] = "/tmp/nova_home_ipid_2024.md"

    with pytest.raises(ValidationError, match="relatif"):
        DocumentCatalog.model_validate(catalog_data)
