"""Post-process and validate source-grounded LLM answers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from app.rag.retriever import RetrievedChunk


_SOURCE_PATTERN = re.compile(r"\[Source\s+(\d+)\]", flags=re.IGNORECASE)


class AnswerProcessingError(ValueError):
    """Raised when a generated answer cannot be safely processed."""


@dataclass(frozen=True, slots=True)
class AnswerCitation:
    """Frontend-ready citation associated with a retrieved chunk."""

    source_number: int
    citation_text: str
    chunk_id: str
    document_id: str
    version_id: str
    title: str | None
    page_number: int | None
    source_path: str | None
    excerpt: str
    relevance_score: float
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class ProcessedAnswer:
    """Validated answer and structured citation information."""

    answer: str
    citations: list[AnswerCitation]
    cited_source_numbers: tuple[int, ...]
    uncited_source_numbers: tuple[int, ...]
    has_valid_citations: bool
    used_fallback: bool


class AnswerProcessor:
    """Validate generated source references and construct citation records."""

    DEFAULT_FALLBACK_ANSWER = (
        "The available indexed documents do not provide enough information "
        "to answer this question reliably."
    )

    def __init__(
        self,
        *,
        require_citations_when_sources_exist: bool = True,
        reject_invalid_citations: bool = True,
        fallback_answer: str | None = None,
        excerpt_character_limit: int = 300,
    ) -> None:
        if excerpt_character_limit < 1:
            raise ValueError("excerpt_character_limit must be at least 1")

        selected_fallback = (
            fallback_answer
            if fallback_answer is not None
            else self.DEFAULT_FALLBACK_ANSWER
        )
        if not selected_fallback.strip():
            raise ValueError("fallback_answer must not be empty")

        self.require_citations_when_sources_exist = require_citations_when_sources_exist
        self.reject_invalid_citations = reject_invalid_citations
        self.fallback_answer = selected_fallback.strip()
        self.excerpt_character_limit = excerpt_character_limit

    def process(
        self,
        *,
        generated_answer: str,
        chunks: Iterable[RetrievedChunk],
    ) -> ProcessedAnswer:
        """Validate an LLM answer against the retrieved source set."""
        answer = generated_answer.strip()
        chunk_list = [chunk for chunk in chunks if chunk.text.strip()]

        if not answer:
            return self._fallback_result()

        cited_numbers = self.extract_source_numbers(answer)
        valid_numbers = set(range(1, len(chunk_list) + 1))
        invalid_numbers = tuple(
            number for number in cited_numbers if number not in valid_numbers
        )

        if invalid_numbers and self.reject_invalid_citations:
            invalid_text = ", ".join(str(number) for number in invalid_numbers)
            raise AnswerProcessingError(
                "The generated answer cites unavailable source numbers: "
                f"{invalid_text}."
            )

        filtered_cited_numbers = tuple(
            number for number in cited_numbers if number in valid_numbers
        )

        if (
            chunk_list
            and self.require_citations_when_sources_exist
            and not filtered_cited_numbers
        ):
            raise AnswerProcessingError(
                "The generated answer does not cite any retrieved sources."
            )

        citations = [
            self._build_citation(
                source_number=number,
                chunk=chunk_list[number - 1],
            )
            for number in filtered_cited_numbers
        ]

        uncited_numbers = tuple(
            number
            for number in sorted(valid_numbers)
            if number not in filtered_cited_numbers
        )

        return ProcessedAnswer(
            answer=answer,
            citations=citations,
            cited_source_numbers=filtered_cited_numbers,
            uncited_source_numbers=uncited_numbers,
            has_valid_citations=bool(citations),
            used_fallback=False,
        )

    @staticmethod
    def extract_source_numbers(answer: str) -> tuple[int, ...]:
        """Return unique source numbers in first-appearance order."""
        seen: set[int] = set()
        numbers: list[int] = []

        for match in _SOURCE_PATTERN.finditer(answer):
            number = int(match.group(1))
            if number in seen:
                continue
            seen.add(number)
            numbers.append(number)

        return tuple(numbers)

    def _build_citation(
        self,
        *,
        source_number: int,
        chunk: RetrievedChunk,
    ) -> AnswerCitation:
        return AnswerCitation(
            source_number=source_number,
            citation_text=f"[Source {source_number}]",
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            title=chunk.title,
            page_number=chunk.page_number,
            source_path=chunk.source_path,
            excerpt=self._make_excerpt(chunk.text),
            relevance_score=chunk.relevance_score,
            metadata=dict(chunk.metadata),
        )

    def _make_excerpt(self, text: str) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= self.excerpt_character_limit:
            return normalized
        shortened = normalized[: self.excerpt_character_limit].rstrip()
        return f"{shortened}..."

    def _fallback_result(self) -> ProcessedAnswer:
        return ProcessedAnswer(
            answer=self.fallback_answer,
            citations=[],
            cited_source_numbers=(),
            uncited_source_numbers=(),
            has_valid_citations=False,
            used_fallback=True,
        )
