import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from nova_assistant.domain import DocumentMetadata
from nova_assistant.ingestion.models import IngestedPage

MULTIPLE_SPACES_PATTERN = re.compile(r"[^\S\n]+")
PAGE_FOOTER_PATTERN = re.compile(r"^Page\s+[1-9][0-9]*$", re.IGNORECASE)
SOURCE_FILENAME_PATTERN = re.compile(r"^nova_(home|auto|travel)_[a-z_]+_[0-9]{4}\.md$")

BOILERPLATE_LINES = {
    "NOVA ASSURANCES",
    "Document fictif - démonstration technique",
}


def normalize_extracted_text(text: str) -> str:
    """Normalise le texte sans modifier son contenu métier."""

    normalized_text = unicodedata.normalize("NFKC", text)
    cleaned_lines: list[str] = []

    for raw_line in normalized_text.splitlines():
        line = MULTIPLE_SPACES_PATTERN.sub(" ", raw_line).strip()

        if not line:
            continue

        if line in BOILERPLATE_LINES:
            continue

        if PAGE_FOOTER_PATTERN.fullmatch(line):
            continue

        if SOURCE_FILENAME_PATTERN.fullmatch(line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_pdf_pages(
    document: DocumentMetadata,
    project_root: Path = Path("."),
) -> tuple[IngestedPage, ...]:
    """Extrait toutes les pages non vides d’un document PDF."""

    pdf_path = project_root / document.pdf_path

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    reader = PdfReader(pdf_path)

    if reader.is_encrypted:
        raise ValueError(f"PDF chiffré non pris en charge : {pdf_path}")

    if not reader.pages:
        raise ValueError(f"PDF sans page : {pdf_path}")

    extracted_pages: list[IngestedPage] = []

    for page_number, pdf_page in enumerate(reader.pages, start=1):
        raw_text = pdf_page.extract_text() or ""
        cleaned_text = normalize_extracted_text(raw_text)

        if not cleaned_text:
            raise ValueError(f"Page vide après extraction : {pdf_path}, page {page_number}")

        extracted_pages.append(
            IngestedPage.from_document(
                document=document,
                page_number=page_number,
                text=cleaned_text,
            )
        )

    return tuple(extracted_pages)
