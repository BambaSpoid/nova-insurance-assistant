from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from nova_assistant.domain import load_catalog
from nova_assistant.ingestion import (
    IngestedPage,
    extract_pdf_pages,
    ingest_catalog,
    load_ingested_pages,
    normalize_extracted_text,
)


def test_normalize_extracted_text_removes_pdf_boilerplate() -> None:
    raw_text = """
    NOVA ASSURANCES
    nova_travel_ipid_2025.md

    Texte    métier   utile.

    Document fictif - démonstration technique
    Page 1
    """

    assert normalize_extracted_text(raw_text) == "Texte métier utile."


def test_ingested_page_has_deterministic_hash() -> None:
    document = load_catalog().documents[0]

    first_page = IngestedPage.from_document(
        document=document,
        page_number=1,
        text="Même contenu",
    )
    second_page = IngestedPage.from_document(
        document=document,
        page_number=1,
        text="Même contenu",
    )

    assert first_page.content_sha256 == second_page.content_sha256
    assert len(first_page.content_sha256) == 64


def test_ingested_page_rejects_blank_text() -> None:
    document = load_catalog().documents[0]

    with pytest.raises(ValidationError, match="vide"):
        IngestedPage.from_document(
            document=document,
            page_number=1,
            text="   ",
        )


def test_extract_pdf_pages_preserves_traceability() -> None:
    catalog = load_catalog()
    document = next(
        document
        for document in catalog.documents
        if document.document_id == "NOVA-TRAVEL-IPID-2025"
    )

    pages = extract_pdf_pages(document)

    assert len(pages) == 2
    assert [page.page_number for page in pages] == [1, 2]
    assert pages[0].page_id == "NOVA-TRAVEL-IPID-2025:page:1"
    assert all(page.document_id == document.document_id for page in pages)
    assert all(page.pdf_path == document.pdf_path for page in pages)
    assert all("NOVA ASSURANCES" not in page.text for page in pages)


def test_ingest_catalog_produces_expected_pages(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "pages.jsonl"

    generated_pages = ingest_catalog(output_path=output_path)
    loaded_pages = load_ingested_pages(output_path)

    assert len(generated_pages) == 31
    assert generated_pages == loaded_pages
    assert len({page.page_id for page in loaded_pages}) == 31
    assert len({page.document_id for page in loaded_pages}) == 11

    product_counts = Counter(page.product.value for page in loaded_pages)
    assert product_counts == {
        "home": 10,
        "auto": 12,
        "travel": 9,
    }


def test_ingestion_is_deterministic(tmp_path: Path) -> None:
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"

    ingest_catalog(output_path=first_output)
    ingest_catalog(output_path=second_output)

    assert first_output.read_bytes() == second_output.read_bytes()


def test_load_ingested_pages_rejects_invalid_jsonl(
    tmp_path: Path,
) -> None:
    invalid_file = tmp_path / "invalid.jsonl"
    invalid_file.write_text("ceci n'est pas du JSON\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Ligne JSONL invalide"):
        load_ingested_pages(invalid_file)
