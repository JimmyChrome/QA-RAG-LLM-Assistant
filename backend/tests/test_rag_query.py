"""Tests for the end-to-end RAG query service."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.rag.answer_processor import (
    AnswerProcessingError,
    AnswerProcessor,
)
from app.rag.llm import GenerationOptions, LLMResponse
from app.rag.prompt_builder import ChatMessage, PromptBuilder
from app.rag.retriever import RetrievalResult, RetrievedChunk
from app.services.rag_query import (
    RAGQueryRequest,
    RAGQueryService,
)


def make_chunk(
    index: int,
    text: str,
    *,
    title: str = "QA Manual",
    page_number: int | None = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{index}",
        text=text,
        document_id=f"document-{index}",
        version_id=f"version-{index}",
        page_number=page_number,
        chunk_index=index - 1,
        distance=0.1,
        relevance_score=0.9,
        title=title,
        source_path=f"/documents/document-{index}.pdf",
        metadata={"office": "QA"},
    )


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "document_ids": document_ids,
                "allowed_version_ids": allowed_version_ids,
                "minimum_relevance_score": minimum_relevance_score,
                "deduplicate": deduplicate,
            }
        )
        return RetrievalResult(
            query=query,
            chunks=list(self.chunks),
            searched_limit=limit or len(self.chunks),
        )


class FakeLLM:
    def __init__(
        self,
        content: str,
        *,
        model: str = "fake-model",
    ) -> None:
        self.content = content
        self.model = model
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        messages,
        *,
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "options": options,
            }
        )
        return LLMResponse(
            content=self.content,
            model=self.model,
            finish_reason="stop",
            prompt_tokens=40,
            completion_tokens=12,
            total_duration_ns=1000,
        )


def make_service(
    *,
    chunks: list[RetrievedChunk],
    llm_answer: str,
) -> tuple[RAGQueryService, FakeRetriever, FakeLLM]:
    retriever = FakeRetriever(chunks)
    llm = FakeLLM(llm_answer)
    service = RAGQueryService(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        llm_provider=llm,
        answer_processor=AnswerProcessor(),
    )
    return service, retriever, llm


def test_complete_query_flow() -> None:
    service, retriever, llm = make_service(
        chunks=[
            make_chunk(
                1,
                "Internal assessment is conducted every semester.",
                page_number=7,
            )
        ],
        llm_answer=(
            "Internal assessment is conducted every semester [Source 1]."
        ),
    )

    result = service.query(
        RAGQueryRequest(
            question="When is internal assessment conducted?",
        )
    )

    assert result.answer.endswith("[Source 1].")
    assert result.question == "When is internal assessment conducted?"
    assert result.model == "fake-model"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 40
    assert result.completion_tokens == 12
    assert result.source_count == 1
    assert result.has_context is True
    assert result.has_valid_citations is True
    assert result.used_fallback is False
    assert len(result.citations) == 1
    assert result.citations[0].page_number == 7
    assert len(result.retrieved_chunks) == 1
    assert retriever.calls[0]["query"] == result.question
    assert len(llm.calls) == 1


def test_retrieval_options_are_forwarded() -> None:
    service, retriever, _ = make_service(
        chunks=[make_chunk(1, "Source text.")],
        llm_answer="Answer [Source 1].",
    )

    options = GenerationOptions(
        temperature=0.1,
        max_tokens=200,
    )
    request = RAGQueryRequest(
        question="Question?",
        limit=8,
        document_ids={"document-1"},
        allowed_version_ids={"version-1"},
        minimum_relevance_score=0.7,
        deduplicate=False,
        generation_options=options,
    )

    service.query(request)

    assert retriever.calls == [
        {
            "query": "Question?",
            "limit": 8,
            "document_ids": {"document-1"},
            "allowed_version_ids": {"version-1"},
            "minimum_relevance_score": 0.7,
            "deduplicate": False,
        }
    ]


def test_generation_options_are_forwarded() -> None:
    service, _, llm = make_service(
        chunks=[make_chunk(1, "Source text.")],
        llm_answer="Answer [Source 1].",
    )
    options = GenerationOptions(
        temperature=0,
        max_tokens=100,
        seed=9,
    )

    service.query(
        RAGQueryRequest(
            question="Question?",
            generation_options=options,
        )
    )

    assert llm.calls[0]["options"] == options


def test_conversation_history_is_forwarded_to_prompt_builder() -> None:
    service, _, llm = make_service(
        chunks=[make_chunk(1, "The next step is review.")],
        llm_answer="The next step is review [Source 1].",
    )

    service.query(
        RAGQueryRequest(
            question="What happens next?",
            conversation_history=(
                ChatMessage(
                    role="user",
                    content="What is the first step?",
                ),
                ChatMessage(
                    role="assistant",
                    content="The first step is submission.",
                ),
            ),
        )
    )

    sent_messages = llm.calls[0]["messages"]
    assert [message.role for message in sent_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_no_context_flow() -> None:
    service, _, llm = make_service(
        chunks=[],
        llm_answer=(
            "The indexed documents do not provide enough information."
        ),
    )

    result = service.query(
        RAGQueryRequest(question="What is the missing deadline?")
    )

    assert result.source_count == 0
    assert result.has_context is False
    assert result.has_valid_citations is False
    assert result.citations == []
    assert "do not provide enough information" in result.answer
    assert "No source excerpts were retrieved" in (
        llm.calls[0]["messages"][-1].content
    )


def test_empty_llm_answer_uses_processor_fallback() -> None:
    service, _, _ = make_service(
        chunks=[],
        llm_answer="   ",
    )

    result = service.query(
        RAGQueryRequest(question="Unknown question?")
    )

    assert result.used_fallback is True
    assert result.has_valid_citations is False
    assert "do not provide enough information" in result.answer


def test_invalid_llm_citation_is_rejected() -> None:
    service, _, _ = make_service(
        chunks=[make_chunk(1, "Only source.")],
        llm_answer="Unsupported answer [Source 4].",
    )

    with pytest.raises(
        AnswerProcessingError,
        match="unavailable source numbers: 4",
    ):
        service.query(RAGQueryRequest(question="Question?"))


def test_uncited_llm_answer_is_rejected() -> None:
    service, _, _ = make_service(
        chunks=[make_chunk(1, "Grounded source.")],
        llm_answer="This answer contains no citation.",
    )

    with pytest.raises(
        AnswerProcessingError,
        match="does not cite any retrieved sources",
    ):
        service.query(RAGQueryRequest(question="Question?"))


@pytest.mark.parametrize("question", ["", " ", "\n\t"])
def test_blank_question_is_rejected(question: str) -> None:
    service, retriever, llm = make_service(
        chunks=[],
        llm_answer="Answer.",
    )

    with pytest.raises(ValueError, match="question must not be empty"):
        service.query(RAGQueryRequest(question=question))

    assert retriever.calls == []
    assert llm.calls == []


def test_question_is_trimmed_before_retrieval_and_response() -> None:
    service, retriever, _ = make_service(
        chunks=[make_chunk(1, "Source.")],
        llm_answer="Answer [Source 1].",
    )

    result = service.query(
        RAGQueryRequest(question="  What is the policy?  ")
    )

    assert result.question == "What is the policy?"
    assert retriever.calls[0]["query"] == "What is the policy?"


def test_multiple_citations_are_returned_as_structured_records() -> None:
    service, _, _ = make_service(
        chunks=[
            make_chunk(1, "First source.", title="Manual", page_number=2),
            make_chunk(2, "Second source.", title="Guide", page_number=9),
        ],
        llm_answer=(
            "The first point is documented [Source 1], "
            "while the second appears elsewhere [Source 2]."
        ),
    )

    result = service.query(
        RAGQueryRequest(question="Compare the requirements.")
    )

    assert [citation.source_number for citation in result.citations] == [1, 2]
    assert [citation.title for citation in result.citations] == [
        "Manual",
        "Guide",
    ]
    assert [citation.page_number for citation in result.citations] == [2, 9]
