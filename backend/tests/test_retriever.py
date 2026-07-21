"""Tests for the RAG retriever layer."""

from __future__ import annotations

from typing import Any

import pytest

from app.rag.retriever import Retriever
from app.rag.vector_store import VectorSearchResult


class FakeVectorStore:
    """Return predefined results and record the requested search."""

    def __init__(self, results: list[VectorSearchResult]) -> None:
        self.results = results
        self.last_query: str | None = None
        self.last_limit: int | None = None
        self.last_where: dict[str, Any] | None = None

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        self.last_query = query
        self.last_limit = limit
        self.last_where = where
        return self.results[:limit]


def result(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "document-1",
    version_id: str = "version-1",
    page_number: int = 1,
    chunk_index: int = 0,
    distance: float | None = 0.1,
    title: str = "QA Manual",
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "document_id": document_id,
            "version_id": version_id,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "title": title,
            "source_path": "/tmp/qa-manual.pdf",
        },
        distance=distance,
    )


def test_retrieve_returns_citation_ready_chunks() -> None:
    store = FakeVectorStore(
        [
            result(
                "chunk-1",
                "Internal assessment is conducted every semester.",
                page_number=7,
                distance=0.08,
            )
        ]
    )
    retriever = Retriever(store)

    output = retriever.retrieve("When is internal assessment conducted?")

    assert len(output.chunks) == 1
    chunk = output.chunks[0]
    assert chunk.document_id == "document-1"
    assert chunk.version_id == "version-1"
    assert chunk.relevance_score == 0.92
    assert chunk.citation_label == "QA Manual, p. 7"
    assert output.context_text == (
        "[Source 1: QA Manual, p. 7]\n"
        "Internal assessment is conducted every semester."
    )


def test_retrieve_applies_relevance_threshold() -> None:
    store = FakeVectorStore(
        [
            result("strong", "Highly relevant text", distance=0.1),
            result("weak", "Weakly relevant text", distance=0.65),
        ]
    )
    retriever = Retriever(
        store,
        minimum_relevance_score=0.5,
    )

    output = retriever.retrieve("assessment")

    assert [chunk.chunk_id for chunk in output.chunks] == ["strong"]


def test_retrieve_deduplicates_normalized_text() -> None:
    store = FakeVectorStore(
        [
            result("chunk-1", "Quality assurance policy", distance=0.05),
            result(
                "chunk-2",
                "  QUALITY   assurance policy  ",
                distance=0.06,
                chunk_index=1,
            ),
            result(
                "chunk-3",
                "Accreditation requirements",
                distance=0.07,
                chunk_index=2,
            ),
        ]
    )
    retriever = Retriever(store)

    output = retriever.retrieve("quality assurance", limit=5)

    assert [chunk.chunk_id for chunk in output.chunks] == [
        "chunk-1",
        "chunk-3",
    ]


def test_retrieve_restricts_active_versions() -> None:
    store = FakeVectorStore(
        [
            result(
                "old",
                "Old version text",
                version_id="version-old",
                distance=0.02,
            ),
            result(
                "active",
                "Active version text",
                version_id="version-active",
                distance=0.05,
            ),
        ]
    )
    retriever = Retriever(store)

    output = retriever.retrieve(
        "policy",
        allowed_version_ids={"version-active"},
    )

    assert [chunk.chunk_id for chunk in output.chunks] == ["active"]
    assert store.last_where == {"version_id": "version-active"}


def test_retrieve_builds_combined_chroma_filter() -> None:
    store = FakeVectorStore(
        [
            result(
                "allowed",
                "Allowed text",
                document_id="document-2",
                version_id="version-2",
            )
        ]
    )
    retriever = Retriever(store)

    output = retriever.retrieve(
        "assessment",
        document_ids={"document-1", "document-2"},
        allowed_version_ids={"version-1", "version-2"},
    )

    assert len(output.chunks) == 1
    assert store.last_where == {
        "$and": [
            {
                "document_id": {
                    "$in": ["document-1", "document-2"]
                }
            },
            {
                "version_id": {
                    "$in": ["version-1", "version-2"]
                }
            },
        ]
    }


def test_retrieve_overfetches_before_post_filtering() -> None:
    store = FakeVectorStore(
        [
            result(
                "wrong-version",
                "Wrong version",
                version_id="version-old",
            ),
            result(
                "right-version",
                "Right version",
                version_id="version-active",
            ),
        ]
    )
    retriever = Retriever(store, overfetch_multiplier=4)

    output = retriever.retrieve(
        "policy",
        limit=1,
        allowed_version_ids={"version-active"},
    )

    assert store.last_limit == 4
    assert [chunk.chunk_id for chunk in output.chunks] == [
        "right-version"
    ]
    assert output.searched_limit == 4


def test_empty_allowed_version_set_returns_without_search() -> None:
    store = FakeVectorStore([result("chunk-1", "Text")])
    retriever = Retriever(store)

    output = retriever.retrieve(
        "policy",
        allowed_version_ids=set(),
    )

    assert output.chunks == []
    assert output.searched_limit == 0
    assert store.last_query is None


def test_missing_document_metadata_is_rejected() -> None:
    broken = VectorSearchResult(
        chunk_id="broken",
        text="Broken result",
        metadata={"page_number": 1},
        distance=0.1,
    )
    retriever = Retriever(FakeVectorStore([broken]))

    with pytest.raises(ValueError, match="missing document metadata"):
        retriever.retrieve("policy")


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.0, 1.0),
        (0.25, 0.75),
        (1.0, 0.0),
        (2.0, 0.0),
        (-0.5, 1.0),
        (None, None),
    ],
)
def test_distance_to_relevance(
    distance: float | None,
    expected: float | None,
) -> None:
    assert Retriever.distance_to_relevance(distance) == expected
