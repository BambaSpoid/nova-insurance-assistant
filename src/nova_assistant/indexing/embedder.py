from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class SentenceTransformerEmbedder:
    """Encode les passages et les requêtes dans le même espace vectoriel."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size doit être strictement positif.")

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)

        dimension = self._model.get_embedding_dimension()

        if dimension is None:
            raise ValueError("La dimension du modèle d’embeddings est inconnue.")

        self.dimension = dimension

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        if not texts:
            raise ValueError("Aucun passage à encoder.")

        prepared_texts = [f"passage: {self._validate_text(text)}" for text in texts]

        embeddings = self._model.encode(
            prepared_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        prepared_query = f"query: {self._validate_text(query)}"

        embedding = self._model.encode(
            [prepared_query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(embedding[0], dtype=np.float32)

    @staticmethod
    def _validate_text(text: str) -> str:
        normalized_text = text.strip()

        if not normalized_text:
            raise ValueError("Le texte à encoder ne doit pas être vide.")

        return normalized_text
