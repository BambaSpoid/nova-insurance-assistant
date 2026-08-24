from pathlib import Path

import numpy as np
import pytest

from nova_assistant.indexing import (
    IndexedChunk,
    VectorIndex,
    chunk_page,
    file_sha256,
    load_semantic_index,
    save_semantic_index,
    validate_chunking_parameters,
)
from nova_assistant.ingestion import load_ingested_pages


def make_sample_chunks() -> tuple[IndexedChunk, ...]:
    pages = load_ingested_pages()
    pages_by_document = {}

    for page in pages:
        pages_by_document.setdefault(page.document_id, page)

    selected_pages = tuple(pages_by_document.values())[:3]
    chunks = []

    for page in selected_pages:
        words = page.text.split()[:20]
        chunks.append(
            IndexedChunk.from_page(
                page=page,
                chunk_number=1,
                text=" ".join(words),
                word_start=0,
                word_end=len(words),
            )
        )

    return tuple(chunks)


def test_chunk_page_creates_expected_overlap() -> None:
    page = load_ingested_pages()[0]

    chunks = chunk_page(
        page=page,
        chunk_size=20,
        chunk_overlap=5,
    )

    assert len(chunks) > 1
    assert chunks[0].word_start == 0
    assert chunks[0].word_end == 20
    assert chunks[1].word_start == 15
    assert chunks[1].word_end == 35

    first_overlap = chunks[0].text.split()[-5:]
    second_overlap = chunks[1].text.split()[:5]

    assert first_overlap == second_overlap


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (20, -1),
        (20, 20),
    ],
)
def test_chunking_rejects_invalid_parameters(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        validate_chunking_parameters(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_vector_index_ranks_by_cosine_similarity() -> None:
    chunks = make_sample_chunks()
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    index = VectorIndex(chunks, embeddings)

    results = index.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == chunks[0].chunk_id
    assert results[0].score == pytest.approx(1.0)
    assert results[1].chunk.chunk_id == chunks[1].chunk_id


def test_vector_index_respects_allowed_documents() -> None:
    chunks = make_sample_chunks()
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    index = VectorIndex(chunks, embeddings)
    allowed_document_id = chunks[2].document_id

    results = index.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=3,
        allowed_document_ids={allowed_document_id},
    )

    assert results
    assert all(result.chunk.document_id == allowed_document_id for result in results)


def test_vector_index_rejects_wrong_query_dimension() -> None:
    chunks = make_sample_chunks()
    embeddings = np.ones((len(chunks), 3), dtype=np.float32)
    index = VectorIndex(chunks, embeddings)

    with pytest.raises(ValueError, match="dimension"):
        index.search(np.asarray([1.0, 0.0], dtype=np.float32))


def test_semantic_index_round_trip(tmp_path: Path) -> None:
    chunks = make_sample_chunks()
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    original_index = VectorIndex(chunks, embeddings)

    original_manifest = save_semantic_index(
        index=original_index,
        model_name="test-model",
        source_pages_path=Path("data/processed/pages.jsonl"),
        index_directory=tmp_path,
        chunk_size=20,
        chunk_overlap=5,
    )
    loaded_index, loaded_manifest = load_semantic_index(tmp_path)

    assert loaded_manifest == original_manifest
    assert loaded_index.chunks == original_index.chunks
    np.testing.assert_allclose(
        loaded_index.embeddings,
        original_index.embeddings,
    )


def test_semantic_index_detects_modified_chunks(
    tmp_path: Path,
) -> None:
    chunks = make_sample_chunks()
    embeddings = np.eye(len(chunks), dtype=np.float32)
    index = VectorIndex(chunks, embeddings)

    manifest = save_semantic_index(
        index=index,
        model_name="test-model",
        source_pages_path=Path("data/processed/pages.jsonl"),
        index_directory=tmp_path,
    )

    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        chunks_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    assert file_sha256(chunks_path) != manifest.chunks_sha256

    with pytest.raises(ValueError, match="passages"):
        load_semantic_index(tmp_path)
