from collections.abc import Collection, Sequence

import numpy as np
from numpy.typing import NDArray

from nova_assistant.indexing.models import (
    IndexedChunk,
    SemanticSearchResult,
)


class VectorIndex:
    """Index vectoriel local utilisant la similarité cosinus."""

    def __init__(
        self,
        chunks: Sequence[IndexedChunk],
        embeddings: NDArray[np.float32],
    ) -> None:
        if not chunks:
            raise ValueError("L’index doit contenir au moins un passage.")

        matrix = np.asarray(embeddings, dtype=np.float32)

        if matrix.ndim != 2:
            raise ValueError("La matrice d’embeddings doit avoir deux dimensions.")

        if matrix.shape[0] != len(chunks):
            raise ValueError("Le nombre de vecteurs doit correspondre aux passages.")

        if matrix.shape[1] == 0:
            raise ValueError("La dimension des embeddings ne peut pas être nulle.")

        if not np.isfinite(matrix).all():
            raise ValueError("La matrice d’embeddings contient des valeurs invalides.")

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("L’index contient des chunk_id dupliqués.")

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)

        if np.any(norms == 0):
            raise ValueError("Un embedding de passage possède une norme nulle.")

        self.chunks = tuple(chunks)
        self.embeddings = matrix / norms
        self.dimension = matrix.shape[1]

    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 5,
        allowed_document_ids: Collection[str] | None = None,
    ) -> tuple[SemanticSearchResult, ...]:
        if top_k <= 0:
            raise ValueError("top_k doit être strictement positif.")

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim != 1 or query.shape[0] != self.dimension:
            raise ValueError("La dimension de la requête ne correspond pas à l’index.")

        if not np.isfinite(query).all():
            raise ValueError("L’embedding de requête contient des valeurs invalides.")

        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            raise ValueError("L’embedding de requête possède une norme nulle.")

        normalized_query = query / query_norm
        candidate_indices = np.arange(len(self.chunks))

        if allowed_document_ids is not None:
            allowed_ids = set(allowed_document_ids)
            candidate_indices = np.asarray(
                [
                    index
                    for index, chunk in enumerate(self.chunks)
                    if chunk.document_id in allowed_ids
                ],
                dtype=np.int64,
            )

        if candidate_indices.size == 0:
            return ()

        candidate_scores = self.embeddings[candidate_indices] @ normalized_query
        ranking = np.argsort(-candidate_scores, kind="stable")
        selected_positions = ranking[:top_k]

        results: list[SemanticSearchResult] = []

        for position in selected_positions:
            chunk_index = int(candidate_indices[position])
            score = float(np.clip(candidate_scores[position], -1.0, 1.0))
            results.append(
                SemanticSearchResult(
                    chunk=self.chunks[chunk_index],
                    score=score,
                )
            )

        return tuple(results)
