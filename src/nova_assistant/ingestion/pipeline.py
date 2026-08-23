import json
from pathlib import Path

from nova_assistant.domain import (
    DEFAULT_CATALOG_PATH,
    DocumentCatalog,
    load_catalog,
)
from nova_assistant.ingestion.models import IngestedPage
from nova_assistant.ingestion.pdf_extractor import extract_pdf_pages

DEFAULT_OUTPUT_PATH = Path("data/processed/pages.jsonl")


def write_ingested_pages(
    pages: tuple[IngestedPage, ...],
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Écrit les pages au format JSONL de façon atomique."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for page in pages:
                serialized_page = json.dumps(
                    page.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                output_file.write(f"{serialized_page}\n")

        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_ingested_pages(
    input_path: Path = DEFAULT_OUTPUT_PATH,
) -> tuple[IngestedPage, ...]:
    """Recharge et valide les pages contenues dans un fichier JSONL."""

    if not input_path.is_file():
        raise FileNotFoundError(f"Fichier d’ingestion introuvable : {input_path}")

    pages: list[IngestedPage] = []

    with input_path.open(encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                page = IngestedPage.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Ligne JSONL invalide {line_number} : {input_path}") from error

            pages.append(page)

    page_ids = [page.page_id for page in pages]

    if len(page_ids) != len(set(page_ids)):
        raise ValueError("Le fichier d’ingestion contient des page_id dupliqués.")

    if not pages:
        raise ValueError("Le fichier d’ingestion ne contient aucune page.")

    return tuple(pages)


def ingest_catalog(
    catalog: DocumentCatalog | None = None,
    project_root: Path = Path("."),
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> tuple[IngestedPage, ...]:
    """Extrait tous les documents recherchables du catalogue."""

    selected_catalog = catalog or load_catalog(project_root / DEFAULT_CATALOG_PATH)
    selected_catalog.validate_files_exist(project_root)

    pages: list[IngestedPage] = []

    for document in selected_catalog.documents:
        if not document.searchable:
            continue

        pages.extend(
            extract_pdf_pages(
                document=document,
                project_root=project_root,
            )
        )

    ingested_pages = tuple(pages)
    page_ids = [page.page_id for page in ingested_pages]

    if len(page_ids) != len(set(page_ids)):
        raise ValueError("L’ingestion a produit des page_id dupliqués.")

    destination = output_path if output_path.is_absolute() else project_root / output_path
    write_ingested_pages(ingested_pages, destination)

    return ingested_pages
