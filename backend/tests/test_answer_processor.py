"""Tests for answer validation and citation extraction."""

from __future__ import annotations

import pytest

from app.rag.answer_processor import AnswerProcessingError, AnswerProcessor
from app.rag.retriever import RetrievedChunk


def chunk(
    source_index: int,
    text: str,
    *,
    page_number: int | None = None,
    title: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{source_index}",
        text=text,
        document_id=f"document-{source_index}",
        version_id=f"version-{source_index}",
        page_number=page_number,
        chunk_index=source_index - 1,
        distance=0.1 * source_index,
        relevance_score=max(0.0, 1.0 - (0.1 * source_index)),
        title=title,
        source_path=f"/documents/source-{source_index}.pdf",
        metadata={"department": "QA Office"},
    )


def test_extracts_valid_citations_in_appearance_order() -> None:
    processor = AnswerProcessor()
    chunks = [
        chunk(1, "First source.", page_number=2, title="Manual"),
        chunk(2, "Second source.", page_number=8, title="Guide"),
    ]
    result = processor.process(
        generated_answer=(
            "The first rule applies [Source 2]. "
            "The supporting definition is in [Source 1]."
        ),
        chunks=chunks,
    )
    assert result.cited_source_numbers == (2, 1)
    assert result.uncited_source_numbers == ()
    assert result.has_valid_citations is True
    assert result.used_fallback is False
    assert [citation.source_number for citation in result.citations] == [2, 1]
    assert result.citations[0].title == "Guide"
    assert result.citations[0].page_number == 8


def test_duplicate_citations_are_returned_once() -> None:
    processor = AnswerProcessor()
    result = processor.process(
        generated_answer="Requirement [Source 1]. Detail [Source 1].",
        chunks=[chunk(1, "Requirement text.")],
    )
    assert result.cited_source_numbers == (1,)
    assert len(result.citations) == 1


def test_source_matching_is_case_insensitive() -> None:
    result = AnswerProcessor().process(
        generated_answer="The answer is documented [source 1].",
        chunks=[chunk(1, "Documented answer.")],
    )
    assert result.cited_source_numbers == (1,)


def test_invalid_source_number_is_rejected() -> None:
    with pytest.raises(AnswerProcessingError, match="unavailable source numbers: 3"):
        AnswerProcessor().process(
            generated_answer="Claim [Source 3].",
            chunks=[chunk(1, "First."), chunk(2, "Second.")],
        )


def test_multiple_invalid_source_numbers_are_reported() -> None:
    with pytest.raises(AnswerProcessingError, match="4, 9"):
        AnswerProcessor().process(
            generated_answer="Claims [Source 4] and [Source 9].",
            chunks=[chunk(1, "Only source.")],
        )


def test_uncited_answer_is_rejected_when_sources_exist() -> None:
    with pytest.raises(AnswerProcessingError, match="does not cite any retrieved sources"):
        AnswerProcessor().process(
            generated_answer="The policy applies every semester.",
            chunks=[chunk(1, "The policy applies every semester.")],
        )


def test_uncited_answer_can_be_allowed() -> None:
    result = AnswerProcessor(require_citations_when_sources_exist=False).process(
        generated_answer="The policy applies every semester.",
        chunks=[chunk(1, "The policy applies every semester.")],
    )
    assert result.citations == []
    assert result.uncited_source_numbers == (1,)


def test_invalid_citations_can_be_ignored() -> None:
    processor = AnswerProcessor(
        reject_invalid_citations=False,
        require_citations_when_sources_exist=False,
    )
    result = processor.process(
        generated_answer="Unsupported [Source 99].",
        chunks=[chunk(1, "Only source.")],
    )
    assert result.citations == []
    assert result.uncited_source_numbers == (1,)


def test_empty_answer_uses_fallback() -> None:
    result = AnswerProcessor().process(
        generated_answer="   ",
        chunks=[chunk(1, "Available source.")],
    )
    assert result.used_fallback is True
    assert result.has_valid_citations is False
    assert "do not provide enough information" in result.answer


def test_empty_answer_uses_custom_fallback() -> None:
    result = AnswerProcessor(
        fallback_answer="No grounded answer is currently available."
    ).process(generated_answer="", chunks=[])
    assert result.answer == "No grounded answer is currently available."


def test_answer_without_sources_is_allowed() -> None:
    result = AnswerProcessor().process(
        generated_answer="The indexed documents do not provide enough information.",
        chunks=[],
    )
    assert result.citations == []
    assert result.used_fallback is False


def test_citation_contains_frontend_ready_metadata() -> None:
    result = AnswerProcessor().process(
        generated_answer="The procedure is listed [Source 1].",
        chunks=[
            chunk(
                1,
                "The procedure is listed in the quality assurance manual.",
                page_number=5,
                title="Quality Assurance Manual",
            )
        ],
    )
    citation = result.citations[0]
    assert citation.citation_text == "[Source 1]"
    assert citation.chunk_id == "chunk-1"
    assert citation.document_id == "document-1"
    assert citation.version_id == "version-1"
    assert citation.title == "Quality Assurance Manual"
    assert citation.page_number == 5
    assert citation.metadata == {"department": "QA Office"}


def test_excerpt_collapses_whitespace() -> None:
    result = AnswerProcessor().process(
        generated_answer="Answer [Source 1].",
        chunks=[chunk(1, "First\n\nsecond\tthird.")],
    )
    assert result.citations[0].excerpt == "First second third."


def test_long_excerpt_is_truncated() -> None:
    result = AnswerProcessor(excerpt_character_limit=20).process(
        generated_answer="Answer [Source 1].",
        chunks=[chunk(1, "A" * 50)],
    )
    assert result.citations[0].excerpt == ("A" * 20) + "..."


def test_empty_chunks_are_not_assigned_source_numbers() -> None:
    result = AnswerProcessor().process(
        generated_answer="Valid answer [Source 1].",
        chunks=[chunk(1, "   "), chunk(2, "Actual source.")],
    )
    assert result.citations[0].chunk_id == "chunk-2"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("No citation.", ()),
        ("[Source 1]", (1,)),
        ("[SOURCE 2] and [source 1]", (2, 1)),
        ("[Source 3] [Source 3] [Source 2]", (3, 2)),
        ("[Source x]", ()),
    ],
)
def test_extract_source_numbers(answer: str, expected: tuple[int, ...]) -> None:
    assert AnswerProcessor.extract_source_numbers(answer) == expected


def test_invalid_excerpt_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="excerpt_character_limit must be at least 1"):
        AnswerProcessor(excerpt_character_limit=0)


def test_blank_custom_fallback_is_rejected() -> None:
    with pytest.raises(ValueError, match="fallback_answer must not be empty"):
        AnswerProcessor(fallback_answer="   ")
