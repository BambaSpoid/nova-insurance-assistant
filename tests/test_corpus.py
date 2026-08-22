import re
from pathlib import Path

from pypdf import PdfReader

SOURCE_ROOT = Path("data/corpus/source")
PDF_ROOT = Path("data/corpus/pdf")

EXPECTED_DOCUMENT_COUNT = 11
IDENTIFIER_PATTERN = re.compile(r"\*\*Identifiant :\*\*\s+([A-Z0-9-]+)")


def get_source_paths() -> list[Path]:
    return sorted(SOURCE_ROOT.rglob("*.md"))


def get_pdf_paths() -> list[Path]:
    return sorted(PDF_ROOT.rglob("*.pdf"))


def test_source_documents_are_complete() -> None:
    source_paths = get_source_paths()

    assert len(source_paths) == EXPECTED_DOCUMENT_COUNT

    for path in source_paths:
        text = path.read_text(encoding="utf-8")

        assert len(text.split()) >= 300
        assert "Document synthétique créé exclusivement" in text


def test_document_identifiers_are_unique() -> None:
    identifiers = []

    for path in get_source_paths():
        text = path.read_text(encoding="utf-8")
        match = IDENTIFIER_PATTERN.search(text)

        assert match is not None, f"Identifiant absent : {path}"
        identifiers.append(match.group(1))

    assert len(identifiers) == EXPECTED_DOCUMENT_COUNT
    assert len(set(identifiers)) == EXPECTED_DOCUMENT_COUNT


def test_2025_documents_do_not_reference_2024() -> None:
    directories = [
        SOURCE_ROOT / "home" / "2025",
        SOURCE_ROOT / "auto" / "2025",
        SOURCE_ROOT / "travel" / "2025",
    ]

    for directory in directories:
        for path in directory.glob("*.md"):
            text = path.read_text(encoding="utf-8")

            assert "2024" not in text
            assert "Archivé" not in text


def test_each_source_has_a_generated_pdf() -> None:
    expected_pdf_paths = {
        path.relative_to(SOURCE_ROOT).with_suffix(".pdf") for path in get_source_paths()
    }
    actual_pdf_paths = {path.relative_to(PDF_ROOT) for path in get_pdf_paths()}

    assert actual_pdf_paths == expected_pdf_paths


def test_generated_pdfs_are_readable() -> None:
    pdf_paths = get_pdf_paths()

    assert len(pdf_paths) == EXPECTED_DOCUMENT_COUNT

    for path in pdf_paths:
        reader = PdfReader(path)
        extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        assert len(reader.pages) >= 1
        assert len(extracted_text.strip()) >= 500
        assert "NOVA ASSURANCES" in extracted_text
        assert "Document fictif - démonstration technique" in extracted_text
