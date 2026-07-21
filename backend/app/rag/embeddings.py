"""Embedding providers used by the RAG pipeline."""

from __future__ import annotations

from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    """Interface required by vector storage and retrieval components."""

    @property
    def dimension(self) -> int:
        """Return the number of values in each embedding vector."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document chunks."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query."""
        ...


class SentenceTransformerEmbeddingProvider:
    """Generate normalized dense embeddings with Sentence Transformers.

    The model is loaded lazily so importing the backend does not immediately
    download or initialize a machine-learning model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model = None
        self._dimension: int | None = None

    @property
    def model(self):
        """Load and cache the configured Sentence Transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            dimension = self.model.get_sentence_embedding_dimension()
            if dimension is None:
                raise RuntimeError(
                    "The embedding model did not report its vector dimension."
                )
            self._dimension = int(dimension)
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []

        vectors = self.model.encode(
            values,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return vectors.astype(float).tolist()

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Query text must not be empty.")

        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return vector.astype(float).tolist()
