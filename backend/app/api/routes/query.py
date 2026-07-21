"""HTTP endpoint for retrieval-augmented document queries."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_rag_query_service
from app.rag.answer_processor import AnswerProcessingError
from app.rag.llm import (
    GenerationOptions,
    LLMConnectionError,
    LLMResponseError,
)
from app.rag.prompt_builder import ChatMessage
from app.schemas.query import (
    AnswerCitationResponse,
    APIErrorResponse,
    RAGQueryRequestSchema,
    RAGQueryResponseSchema,
)
from app.services.rag_query import (
    RAGQueryRequest,
    RAGQueryService,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["query"],
)


@router.post(
    "/query",
    response_model=RAGQueryResponseSchema,
    responses={
        422: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
        503: {"model": APIErrorResponse},
    },
)
def query_documents(
    payload: RAGQueryRequestSchema,
    service: RAGQueryService = Depends(get_rag_query_service),
) -> RAGQueryResponseSchema:
    """Answer a question using indexed document sources."""
    request = RAGQueryRequest(
        question=payload.question,
        limit=payload.limit,
        document_ids=payload.document_ids,
        allowed_version_ids=payload.allowed_version_ids,
        minimum_relevance_score=payload.minimum_relevance_score,
        deduplicate=payload.deduplicate,
        conversation_history=tuple(
            ChatMessage(
                role=message.role,
                content=message.content,
            )
            for message in payload.conversation_history
        ),
        generation_options=_map_generation_options(
            payload.generation_options
        ),
    )

    try:
        result = service.query(request)
    except AnswerProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "detail": str(exc),
                "code": "answer_validation_failed",
            },
        ) from exc
    except LLMConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "detail": str(exc),
                "code": "llm_unavailable",
            },
        ) from exc
    except LLMResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "detail": str(exc),
                "code": "llm_response_failed",
            },
        ) from exc

    return RAGQueryResponseSchema(
        question=result.question,
        answer=result.answer,
        citations=[
            AnswerCitationResponse(
                source_number=citation.source_number,
                citation_text=citation.citation_text,
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                version_id=citation.version_id,
                title=citation.title,
                page_number=citation.page_number,
                source_path=citation.source_path,
                excerpt=citation.excerpt,
                relevance_score=citation.relevance_score,
                metadata=citation.metadata,
            )
            for citation in result.citations
        ],
        model=result.model,
        finish_reason=result.finish_reason,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        source_count=result.source_count,
        has_context=result.has_context,
        has_valid_citations=result.has_valid_citations,
        used_fallback=result.used_fallback,
    )


def _map_generation_options(options):
    if options is None:
        return None

    return GenerationOptions(
        temperature=options.temperature,
        top_p=options.top_p,
        max_tokens=options.max_tokens,
        seed=options.seed,
        stop=options.stop,
    )
