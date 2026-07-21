"""End-to-end orchestration for retrieval-augmented generation queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from app.rag.answer_processor import (
    AnswerCitation,
    AnswerProcessor,
    ProcessedAnswer,
)
from app.rag.llm import GenerationOptions, LLMProvider, LLMResponse
from app.rag.prompt_builder import ChatMessage, PromptBuilder, PromptPackage
from app.rag.retriever import RetrievalResult, RetrievedChunk


class RetrieverProtocol(Protocol):
    """Minimal retriever interface required by the query service."""

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
        """Retrieve relevant document chunks."""
        ...


@dataclass(frozen=True, slots=True)
class RAGQueryRequest:
    """Input accepted by the RAG query service."""

    question: str
    limit: int | None = None
    document_ids: set[str] | None = None
    allowed_version_ids: set[str] | None = None
    minimum_relevance_score: float | None = None
    deduplicate: bool = True
    conversation_history: tuple[ChatMessage, ...] = ()
    generation_options: GenerationOptions | None = None


@dataclass(frozen=True, slots=True)
class RAGQueryResponse:
    """Complete result of one RAG query."""

    question: str
    answer: str
    citations: list[AnswerCitation]
    retrieved_chunks: list[RetrievedChunk]
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    source_count: int
    has_context: bool
    has_valid_citations: bool
    used_fallback: bool


class RAGQueryService:
    """Coordinate retrieval, prompt construction, generation, and validation."""

    def __init__(
        self,
        *,
        retriever: RetrieverProtocol,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        answer_processor: AnswerProcessor,
    ) -> None:
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm_provider = llm_provider
        self.answer_processor = answer_processor

    def query(self, request: RAGQueryRequest) -> RAGQueryResponse:
        """Execute a complete retrieval-augmented generation request."""
        question = request.question.strip()
        if not question:
            raise ValueError("question must not be empty")

        retrieval = self.retriever.retrieve(
            question,
            limit=request.limit,
            document_ids=request.document_ids,
            allowed_version_ids=request.allowed_version_ids,
            minimum_relevance_score=request.minimum_relevance_score,
            deduplicate=request.deduplicate,
        )

        prompt = self.prompt_builder.build(
            question=question,
            chunks=retrieval.chunks,
            conversation_history=request.conversation_history,
        )

        llm_response = self.llm_provider.generate(
            prompt.messages,
            options=request.generation_options,
        )

        processed = self.answer_processor.process(
            generated_answer=llm_response.content,
            chunks=retrieval.chunks,
        )

        return self._build_response(
            question=question,
            retrieval=retrieval,
            prompt=prompt,
            llm_response=llm_response,
            processed=processed,
        )

    @staticmethod
    def _build_response(
        *,
        question: str,
        retrieval: RetrievalResult,
        prompt: PromptPackage,
        llm_response: LLMResponse,
        processed: ProcessedAnswer,
    ) -> RAGQueryResponse:
        return RAGQueryResponse(
            question=question,
            answer=processed.answer,
            citations=processed.citations,
            retrieved_chunks=retrieval.chunks,
            model=llm_response.model,
            finish_reason=llm_response.finish_reason,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            source_count=prompt.source_count,
            has_context=prompt.has_context,
            has_valid_citations=processed.has_valid_citations,
            used_fallback=processed.used_fallback,
        )
