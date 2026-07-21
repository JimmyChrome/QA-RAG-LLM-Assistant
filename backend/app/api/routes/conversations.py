"""HTTP endpoints for persistent chatbot conversations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app.api.dependencies import get_conversation_service
from app.rag.llm import GenerationOptions
from app.schemas.conversation import (
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationQueryRequest,
    ConversationQueryResponse,
    ConversationResponse,
)
from app.schemas.query import AnswerCitationResponse, RAGQueryResponseSchema
from app.services.conversation import (
    ConversationNotFoundError,
    ConversationResult,
    ConversationService,
)
from app.services.rag_query import RAGQueryRequest


router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["conversations"],
)


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    payload: ConversationCreateRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    return _map_conversation(
        service.create_conversation(payload.title)
    )


@router.get("", response_model=ConversationListResponse)
def list_conversations(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationListResponse:
    conversations = service.list_conversations(limit=limit, offset=offset)
    items = [_map_conversation(item) for item in conversations]
    return ConversationListResponse(items=items, count=len(items))


@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    try:
        return _map_conversation(
            service.get_conversation(conversation_id)
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
) -> Response:
    try:
        service.delete_conversation(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{conversation_id}/query",
    response_model=ConversationQueryResponse,
)
def query_conversation(
    conversation_id: str,
    payload: ConversationQueryRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationQueryResponse:
    request = RAGQueryRequest(
        question=payload.question,
        limit=payload.limit,
        document_ids=payload.document_ids,
        allowed_version_ids=payload.allowed_version_ids,
        minimum_relevance_score=payload.minimum_relevance_score,
        deduplicate=payload.deduplicate,
        generation_options=_map_generation_options(
            payload.generation_options
        ),
    )

    try:
        result = service.query_conversation(
            conversation_id=conversation_id,
            request=request,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    query = result.query
    return ConversationQueryResponse(
        conversation=_map_conversation(result.conversation),
        query=RAGQueryResponseSchema(
            question=query.question,
            answer=query.answer,
            citations=[
                AnswerCitationResponse(
                    source_number=item.source_number,
                    citation_text=item.citation_text,
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    version_id=item.version_id,
                    title=item.title,
                    page_number=item.page_number,
                    source_path=item.source_path,
                    excerpt=item.excerpt,
                    relevance_score=item.relevance_score,
                    metadata=item.metadata,
                )
                for item in query.citations
            ],
            model=query.model,
            finish_reason=query.finish_reason,
            prompt_tokens=query.prompt_tokens,
            completion_tokens=query.completion_tokens,
            source_count=query.source_count,
            has_context=query.has_context,
            has_valid_citations=query.has_valid_citations,
            used_fallback=query.used_fallback,
        ),
    )


def _map_conversation(
    conversation: ConversationResult,
) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "citations": list(message.citations),
                "sequence_number": message.sequence_number,
                "created_at": message.created_at,
            }
            for message in conversation.messages
        ],
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
