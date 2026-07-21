"""Persistent ChromaDB storage for document chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import chromadb

from app.rag.chunker import TextChunk
from app.rag.embeddings import EmbeddingProvider


MetadataValue = str | int | float | bool


@dataclass(slots=True)
class VectorSearchResult:
    """One result returned by vector similarity search."""

    chunk_id: str
    text: str
    metadata: dict[str, MetadataValue]
    distance: float | None


class ChromaVectorStore:
    """Store and search externally generated embeddings in ChromaDB."""

    def __init__(
        self,
        *,
        persist_directory: str | Path,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "qa_document_chunks",
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_provider = embedding_provider

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: Iterable[TextChunk],
        *,
        document_id: str,
        version_id: str,
    ) -> list[str]:
        """Embed and upsert chunks, returning their stable vector IDs."""
        chunk_list = list(chunks)
        if not chunk_list:
            return []
        if not document_id.strip() or not version_id.strip():
            raise ValueError("document_id and version_id are required")

        texts = [chunk.text for chunk in chunk_list]
        embeddings = self.embedding_provider.embed_documents(texts)
        self._validate_embeddings(embeddings, expected_count=len(chunk_list))

        ids = [
            self.build_chunk_id(
                document_id=document_id,
                version_id=version_id,
                chunk_index=chunk.chunk_index,
            )
            for chunk in chunk_list
        ]

        metadatas = [
            self._build_metadata(
                chunk,
                document_id=document_id,
                version_id=version_id,
            )
            for chunk in chunk_list
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return ids

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Return chunks closest to the query embedding."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        query_embedding = self.embedding_provider.embed_query(query)
        if len(query_embedding) != self.embedding_provider.dimension:
            raise ValueError("Query embedding dimension is incorrect.")

        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results: list[VectorSearchResult] = []
        for index, chunk_id in enumerate(ids):
            results.append(
                VectorSearchResult(
                    chunk_id=chunk_id,
                    text=documents[index],
                    metadata=dict(metadatas[index] or {}),
                    distance=(
                        float(distances[index])
                        if index < len(distances)
                        else None
                    ),
                )
            )
        return results

    def delete_version(self, version_id: str) -> None:
        """Delete every vector belonging to one document version."""
        if not version_id.strip():
            raise ValueError("version_id must not be empty")
        self.collection.delete(where={"version_id": version_id})

    def delete_document(self, document_id: str) -> None:
        """Delete every vector belonging to one logical document."""
        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        self.collection.delete(where={"document_id": document_id})

    def count(self) -> int:
        """Return the number of vectors in the collection."""
        return self.collection.count()

    @staticmethod
    def build_chunk_id(
        *,
        document_id: str,
        version_id: str,
        chunk_index: int,
    ) -> str:
        return f"{document_id}:{version_id}:{chunk_index}"

    def _validate_embeddings(
        self,
        embeddings: list[list[float]],
        *,
        expected_count: int,
    ) -> None:
        if len(embeddings) != expected_count:
            raise ValueError(
                "Embedding provider returned an unexpected vector count."
            )

        expected_dimension = self.embedding_provider.dimension
        if any(len(vector) != expected_dimension for vector in embeddings):
            raise ValueError(
                "Embedding provider returned an unexpected vector dimension."
            )

    @staticmethod
    def _build_metadata(
        chunk: TextChunk,
        *,
        document_id: str,
        version_id: str,
    ) -> dict[str, MetadataValue]:
        metadata: dict[str, MetadataValue] = {
            "document_id": document_id,
            "version_id": version_id,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }

        for key, value in chunk.metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value

        return metadata
