import argparse
from collections import Counter
from pathlib import Path

from nova_assistant.ingestion import (
    DEFAULT_OUTPUT_PATH,
    ingest_catalog,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingère les PDF du catalogue documentaire Nova.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Racine du projet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Chemin du fichier JSONL produit.",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()

    pages = ingest_catalog(
        project_root=arguments.project_root,
        output_path=arguments.output,
    )

    product_counts = Counter(page.product.value for page in pages)
    document_count = len({page.document_id for page in pages})

    print("Ingestion terminée.")
    print(f"- Documents : {document_count}")
    print(f"- Pages : {len(pages)}")
    print(f"- Répartition : {dict(product_counts)}")
    print(f"- Sortie : {arguments.output}")


if __name__ == "__main__":
    main()
