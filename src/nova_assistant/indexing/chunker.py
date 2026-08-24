from collections.abc import Iterable

from nova_assistant.indexing.models import IndexedChunk
from nova_assistant.ingestion import IngestedPage

DEFAULT_CHUNK_SIZE = 120
DEFAULT_CHUNK_OVERLAP = 30


def validate_chunking_parameters(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size doit être strictement positif.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap ne peut pas être négatif.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size.")


def chunk_page(
    page: IngestedPage,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[IndexedChunk, ...]:
    """Découpe une page en fenêtres de mots avec chevauchement."""

    validate_chunking_parameters(chunk_size, chunk_overlap)

    words = page.text.split()
    chunks: list[IndexedChunk] = []
    word_start = 0
    chunk_number = 1

    while word_start < len(words):
        word_end = min(word_start + chunk_size, len(words))
        chunk_text = " ".join(words[word_start:word_end])

        chunks.append(
            IndexedChunk.from_page(
                page=page,
                chunk_number=chunk_number,
                text=chunk_text,
                word_start=word_start,
                word_end=word_end,
            )
        )

        if word_end == len(words):
            break

        word_start = word_end - chunk_overlap
        chunk_number += 1

    return tuple(chunks)


def chunk_pages(
    pages: Iterable[IngestedPage],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[IndexedChunk, ...]:
    """Découpe un ensemble de pages et vérifie les identifiants."""

    chunks: list[IndexedChunk] = []

    for page in pages:
        chunks.extend(
            chunk_page(
                page=page,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Le découpage a produit des chunk_id dupliqués.")

    return tuple(chunks)
