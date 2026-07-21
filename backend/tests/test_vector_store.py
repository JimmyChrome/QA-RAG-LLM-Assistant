"""Tests for embedding interfaces and ChromaDB storage."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.rag.chunker import TextChunk
from app.rag.vector_store import ChromaVectorStore


class FakeEmbeddingProvider:
    """Small deterministic embedder used to keep tests offline."""

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Query text must not be empty.")
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("assessment")),
            float(lowered.count("accreditation")),
            float(lowered.count("policy")),
        ]


def build_chunk(
    index: int,
    text: str,
    *,
    page_number: int = 1,
) -> TextChunk:
    return TextChunk(
        chunk_index=index,
        page_number=page_number,
        text=text,
        start_char=index * 100,
        end_char=(index * 100) + len(text),
        metadata={
            "title": "QA Manual",
            "source_path": "/tmp/qa-manual.pdf",
            "ignored_none": None,
            "ignored_list": ["not", "valid", "chroma", "metadata"],
        },
    )


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        embedding_provider=FakeEmbeddingProvider(),
        collection_name="test_chunks",
    )


def test_add_chunks_and_count(store: ChromaVectorStore) -> None:
    ids = store.add_chunks(
        [
            build_chunk(0, "Internal assessment procedures", page_number=2),
            build_chunk(1, "Accreditation evidence requirements", page_number=5),
        ],
        document_id="document-1",
        version_id="version-1",
    )

    assert ids == [
        "document-1:version-1:0",
        "document-1:version-1:1",
    ]
    assert store.count() == 2


def test_search_returns_most_relevant_chunk(store: ChromaVectorStore) -> None:
    store.add_chunks(
        [
            build_chunk(0, "Internal assessment procedures", page_number=2),
            build_chunk(1, "Accreditation evidence requirements", page_number=5),
            build_chunk(2, "Document retention policy", page_number=8),
        ],
        document_id="document-1",
        version_id="version-1",
    )

    results = store.search("What is required for accreditation?", limit=2)

    assert len(results) == 2
    assert results[0].text == "Accreditation evidence requirements"
    assert results[0].metadata["document_id"] == "document-1"
    assert results[0].metadata["version_id"] == "version-1"
    assert results[0].metadata["page_number"] == 5
    assert "ignored_none" not in results[0].metadata
    assert "ignored_list" not in results[0].metadata


def test_search_can_filter_by_document(store: ChromaVectorStore) -> None:
    store.add_chunks(
        [build_chunk(0, "Accreditation requirements")],
        document_id="document-1",
        version_id="version-1",
    )
    store.add_chunks(
        [build_chunk(0, "Accreditation timeline")],
        document_id="document-2",
        version_id="version-2",
    )

    results = store.search(
        "accreditation",
        limit=5,
        where={"document_id": "document-2"},
    )

    assert len(results) == 1
    assert results[0].metadata["document_id"] == "document-2"


def test_upsert_replaces_existing_chunk(store: ChromaVectorStore) -> None:
    store.add_chunks(
        [build_chunk(0, "Old policy text")],
        document_id="document-1",
        version_id="version-1",
    )
    store.add_chunks(
        [build_chunk(0, "Updated policy text")],
        document_id="document-1",
        version_id="version-1",
    )

    assert store.count() == 1
    results = store.search("policy", limit=1)
    assert results[0].text == "Updated policy text"


def test_delete_version(store: ChromaVectorStore) -> None:
    store.add_chunks(
        [build_chunk(0, "First version policy")],
        document_id="document-1",
        version_id="version-1",
    )
    store.add_chunks(
        [build_chunk(0, "Second version policy")],
        document_id="document-1",
        version_id="version-2",
    )

    store.delete_version("version-1")

    assert store.count() == 1
    remaining = store.search("policy", limit=5)
    assert remaining[0].metadata["version_id"] == "version-2"


def test_delete_document(store: ChromaVectorStore) -> None:
    store.add_chunks(
        [build_chunk(0, "Document one policy")],
        document_id="document-1",
        version_id="version-1",
    )
    store.add_chunks(
        [build_chunk(0, "Document two policy")],
        document_id="document-2",
        version_id="version-2",
    )

    store.delete_document("document-1")

    assert store.count() == 1
    remaining = store.search("policy", limit=5)
    assert remaining[0].metadata["document_id"] == "document-2"


def test_reject_invalid_embedding_count(
    tmp_path: Path,
) -> None:
    class BrokenProvider(FakeEmbeddingProvider):
        def embed_documents(
            self,
            texts: Sequence[str],
        ) -> list[list[float]]:
            return []

    store = ChromaVectorStore(
        persist_directory=tmp_path / "broken",
        embedding_provider=BrokenProvider(),
        collection_name="broken_chunks",
    )

    with pytest.raises(ValueError, match="vector count"):
        store.add_chunks(
            [build_chunk(0, "Policy")],
            document_id="document-1",
            version_id="version-1",
        )
