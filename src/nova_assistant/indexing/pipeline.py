import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from nova_assistant.indexing.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_pages,
)
from nova_assistant.indexing.embedder import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
)
from nova_assistant.indexing.models import (
    IndexedChunk,
    IndexManifest,
)
from nova_assistant.indexing.vector_index import VectorIndex
from nova_assistant.ingestion import (
    DEFAULT_OUTPUT_PATH,
    load_ingested_pages,
)

DEFAULT_INDEX_DIRECTORY = Path("data/index")
CHUNKS_FILENAME = "chunks.jsonl"
EMBEDDINGS_FILENAME = "embeddings.npy"
MANIFEST_FILENAME = "manifest.json"


def file_sha256(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def save_semantic_index(
    index: VectorIndex,
    model_name: str,
    source_pages_path: Path = DEFAULT_OUTPUT_PATH,
    index_directory: Path = DEFAULT_INDEX_DIRECTORY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IndexManifest:
    index_directory.mkdir(parents=True, exist_ok=True)

    chunks_path = index_directory / CHUNKS_FILENAME
    embeddings_path = index_directory / EMBEDDINGS_FILENAME
    manifest_path = index_directory / MANIFEST_FILENAME

    temporary_chunks_path = chunks_path.with_suffix(".jsonl.tmp")
    temporary_embeddings_path = embeddings_path.with_suffix(".npy.tmp")
    temporary_manifest_path = manifest_path.with_suffix(".json.tmp")

    temporary_paths = (
        temporary_chunks_path,
        temporary_embeddings_path,
        temporary_manifest_path,
    )

    try:
        with temporary_chunks_path.open(
            "w",
            encoding="utf-8",
        ) as chunks_file:
            for chunk in index.chunks:
                serialized_chunk = json.dumps(
                    chunk.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                chunks_file.write(f"{serialized_chunk}\n")

        with temporary_embeddings_path.open("wb") as embeddings_file:
            np.save(
                embeddings_file,
                index.embeddings,
                allow_pickle=False,
            )

        manifest = IndexManifest(
            model_name=model_name,
            dimension=index.dimension,
            chunk_count=len(index.chunks),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_pages_sha256=file_sha256(source_pages_path),
            chunks_sha256=file_sha256(temporary_chunks_path),
            embeddings_sha256=file_sha256(temporary_embeddings_path),
        )

        temporary_manifest_path.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

        temporary_chunks_path.replace(chunks_path)
        temporary_embeddings_path.replace(embeddings_path)
        temporary_manifest_path.replace(manifest_path)

        return manifest
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def load_semantic_index(
    index_directory: Path = DEFAULT_INDEX_DIRECTORY,
) -> tuple[VectorIndex, IndexManifest]:
    chunks_path = index_directory / CHUNKS_FILENAME
    embeddings_path = index_directory / EMBEDDINGS_FILENAME
    manifest_path = index_directory / MANIFEST_FILENAME

    for required_path in (
        chunks_path,
        embeddings_path,
        manifest_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(f"Fichier d’index introuvable : {required_path}")

    manifest = IndexManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    if file_sha256(chunks_path) != manifest.chunks_sha256:
        raise ValueError("L’empreinte des passages est invalide.")

    if file_sha256(embeddings_path) != manifest.embeddings_sha256:
        raise ValueError("L’empreinte des embeddings est invalide.")

    chunks: list[IndexedChunk] = []

    with chunks_path.open(encoding="utf-8") as chunks_file:
        for line_number, raw_line in enumerate(chunks_file, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                chunk = IndexedChunk.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Passage JSONL invalide à la ligne {line_number}.") from error

            chunks.append(chunk)

    embeddings = np.load(embeddings_path, allow_pickle=False)
    index = VectorIndex(chunks, embeddings)

    if len(index.chunks) != manifest.chunk_count:
        raise ValueError("Le nombre de passages ne correspond pas au manifeste.")

    if index.dimension != manifest.dimension:
        raise ValueError("La dimension de l’index ne correspond pas au manifeste.")

    return index, manifest


def build_semantic_index(
    pages_path: Path = DEFAULT_OUTPUT_PATH,
    index_directory: Path = DEFAULT_INDEX_DIRECTORY,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[VectorIndex, IndexManifest]:
    pages = load_ingested_pages(pages_path)
    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    embedder = SentenceTransformerEmbedder(model_name=model_name)
    embeddings = embedder.embed_passages([chunk.text for chunk in chunks])
    index = VectorIndex(chunks, embeddings)

    manifest = save_semantic_index(
        index=index,
        model_name=model_name,
        source_pages_path=pages_path,
        index_directory=index_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return index, manifest
