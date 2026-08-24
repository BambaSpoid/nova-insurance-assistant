import argparse
from pathlib import Path

from nova_assistant.indexing import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_DIRECTORY,
    build_semantic_index,
)
from nova_assistant.ingestion import DEFAULT_OUTPUT_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construit l’index sémantique Nova.")
    parser.add_argument(
        "--pages",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Fichier JSONL contenant les pages ingérées.",
    )
    parser.add_argument(
        "--index-directory",
        type=Path,
        default=DEFAULT_INDEX_DIRECTORY,
        help="Répertoire de sortie de l’index.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Modèle Sentence Transformers.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Nombre maximal de mots par passage.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Nombre de mots communs entre deux passages.",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()

    index, manifest = build_semantic_index(
        pages_path=arguments.pages,
        index_directory=arguments.index_directory,
        model_name=arguments.model,
        chunk_size=arguments.chunk_size,
        chunk_overlap=arguments.chunk_overlap,
    )

    print("Index sémantique construit.")
    print(f"- Modèle : {manifest.model_name}")
    print(f"- Passages : {len(index.chunks)}")
    print(f"- Dimension : {index.dimension}")
    print(f"- Découpage : {manifest.chunk_size} mots, chevauchement {manifest.chunk_overlap}")
    print(f"- Sortie : {arguments.index_directory}")


if __name__ == "__main__":
    main()
