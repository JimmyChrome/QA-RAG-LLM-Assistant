"""Retrieval layer for citation-ready RAG context."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.rag.vector_store import VectorSearchResult


class VectorSearchStore(Protocol):
    """Minimal vector-store interface required by the retriever."""

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Return vector search results."""
        ...


@dataclass(slots=True)
class RetrievedChunk:
    """One validated and citation-ready retrieved chunk."""

    chunk_id: str
    text: str
    document_id: str
    version_id: str
    page_number: int | None
    chunk_index: int | None
    distance: float | None
    relevance_score: float | None
    title: str | None = None
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        """Return a readable source label for prompts and answers."""
        source = self.title or self.source_path or self.document_id
        if self.page_number is not None:
            return f"{source}, p. {self.page_number}"
        return source


@dataclass(slots=True)
class RetrievalResult:
    """Complete output from one retrieval request."""

    query: str
    chunks: list[RetrievedChunk]
    searched_limit: int

    @property
    def context_text(self) -> str:
        """Format retrieved chunks for prompt construction."""
        sections: list[str] = []
        for index, chunk in enumerate(self.chunks, start=1):
            sections.append(
                f"[Source {index}: {chunk.citation_label}]\n{chunk.text.strip()}"
            )
        return "\n\n".join(sections)


class Retriever:
    """Apply semantic search, restrictions, thresholds, and deduplication."""

    def __init__(
        self,
        vector_store: VectorSearchStore,
        *,
        default_limit: int = 5,
        minimum_relevance_score: float | None = None,
        overfetch_multiplier: int = 4,
    ) -> None:
        if default_limit < 1:
            raise ValueError("default_limit must be at least 1")
        if overfetch_multiplier < 1:
            raise ValueError("overfetch_multiplier must be at least 1")
        if (
            minimum_relevance_score is not None
            and not 0.0 <= minimum_relevance_score <= 1.0
        ):
            raise ValueError(
                "minimum_relevance_score must be between 0 and 1"
            )

        self.vector_store = vector_store
        self.default_limit = default_limit
        self.minimum_relevance_score = minimum_relevance_score
        self.overfetch_multiplier = overfetch_multiplier

    def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        document_ids: set[str] | None = None,
        allowed_version_ids: set[str] | None = None,
        minimum_relevance_score: float | None = None,
        deduplicate: bool = True,
    ) -> RetrievalResult:
        """Retrieve chunks that satisfy all requested restrictions."""
        if not query.strip():
            raise ValueError("query must not be empty")

        final_limit = limit or self.default_limit
        if final_limit < 1:
            raise ValueError("limit must be at least 1")

        threshold = (
            self.minimum_relevance_score
            if minimum_relevance_score is None
            else minimum_relevance_score
        )
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "minimum_relevance_score must be between 0 and 1"
            )

        if document_ids is not None and not document_ids:
            return RetrievalResult(
                query=query,
                chunks=[],
                searched_limit=0,
            )
        if allowed_version_ids is not None and not allowed_version_ids:
            return RetrievalResult(
                query=query,
                chunks=[],
                searched_limit=0,
            )

        search_limit = final_limit * self.overfetch_multiplier
        where = self._build_where_filter(
            document_ids=document_ids,
            allowed_version_ids=allowed_version_ids,
        )

        raw_results = self.vector_store.search(
            query,
            limit=search_limit,
            where=where,
        )

        selected: list[RetrievedChunk] = []
        seen_text: set[str] = set()
        seen_ids: set[str] = set()

        for result in raw_results:
            chunk = self._convert_result(result)

            if document_ids is not None:
                if chunk.document_id not in document_ids:
                    continue

            if allowed_version_ids is not None:
                if chunk.version_id not in allowed_version_ids:
                    continue

            if (
                threshold is not None
                and (
                    chunk.relevance_score is None
                    or chunk.relevance_score < threshold
                )
            ):
                continue

            normalized = self._normalize_for_deduplication(chunk.text)
            if deduplicate and (
                chunk.chunk_id in seen_ids or normalized in seen_text
            ):
                continue

            seen_ids.add(chunk.chunk_id)
            seen_text.add(normalized)
            selected.append(chunk)

            if len(selected) >= final_limit:
                break

        return RetrievalResult(
            query=query,
            chunks=selected,
            searched_limit=search_limit,
        )

    @staticmethod
    def _build_where_filter(
        *,
        document_ids: set[str] | None,
        allowed_version_ids: set[str] | None,
    ) -> dict[str, Any] | None:
        filters: list[dict[str, Any]] = []

        if document_ids:
            values = sorted(document_ids)
            filters.append(
                {"document_id": values[0]}
                if len(values) == 1
                else {"document_id": {"$in": values}}
            )

        if allowed_version_ids:
            values = sorted(allowed_version_ids)
            filters.append(
                {"version_id": values[0]}
                if len(values) == 1
                else {"version_id": {"$in": values}}
            )

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    @staticmethod
    def _convert_result(result: VectorSearchResult) -> RetrievedChunk:
        metadata = dict(result.metadata)

        document_id = str(metadata.get("document_id", "")).strip()
        version_id = str(metadata.get("version_id", "")).strip()
        if not document_id or not version_id:
            raise ValueError(
                f"Vector result '{result.chunk_id}' is missing document metadata."
            )

        page_number = Retriever._optional_int(metadata.get("page_number"))
        chunk_index = Retriever._optional_int(metadata.get("chunk_index"))

        title = Retriever._optional_string(metadata.get("title"))
        source_path = Retriever._optional_string(
            metadata.get("source_path")
        )

        relevance_score = Retriever.distance_to_relevance(result.distance)

        return RetrievedChunk(
            chunk_id=result.chunk_id,
            text=result.text,
            document_id=document_id,
            version_id=version_id,
            page_number=page_number,
            chunk_index=chunk_index,
            distance=result.distance,
            relevance_score=relevance_score,
            title=title,
            source_path=source_path,
            metadata=metadata,
        )

    @staticmethod
    def distance_to_relevance(distance: float | None) -> float | None:
        """Convert cosine distance into a bounded similarity score."""
        if distance is None:
            return None
        return round(max(0.0, min(1.0, 1.0 - float(distance))), 6)

    @staticmethod
    def _normalize_for_deduplication(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().casefold()

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
