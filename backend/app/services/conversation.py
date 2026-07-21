"""Application service for persistent multi-turn conversations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

from app.rag.prompt_builder import ChatMessage
from app.services.rag_query import RAGQueryRequest, RAGQueryResponse, RAGQueryService


class ConversationNotFoundError(LookupError):
    """Raised when a requested conversation does not exist."""


class ConversationRepositoryProtocol(Protocol):
    """Storage operations required by ConversationService."""

    def create(self, *, title: str): ...
    def get(self, conversation_id: str): ...
    def list(self, *, limit: int = 50, offset: int = 0) -> Sequence: ...
    def add_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[dict] | None = None,
    ): ...
    def delete(self, conversation_id: str) -> bool: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ConversationMessageResult:
    id: str
    role: str
    content: str
    citations: tuple[dict, ...]
    sequence_number: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ConversationResult:
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: tuple[ConversationMessageResult, ...] = ()


@dataclass(frozen=True, slots=True)
class ConversationQueryResult:
    conversation: ConversationResult
    query: RAGQueryResponse


class ConversationService:
    """Coordinate conversation persistence with the existing RAG service."""

    def __init__(
        self,
        *,
        repository: ConversationRepositoryProtocol,
        rag_query_service: RAGQueryService,
        history_limit: int = 20,
    ) -> None:
        if history_limit < 1:
            raise ValueError("history_limit must be at least 1")
        self._repository = repository
        self._rag_query_service = rag_query_service
        self._history_limit = history_limit

    def create_conversation(self, title: str | None = None) -> ConversationResult:
        clean_title = (title or "New conversation").strip()
        if not clean_title:
            clean_title = "New conversation"

        try:
            conversation = self._repository.create(title=clean_title[:200])
            self._repository.commit()
        except Exception:
            self._repository.rollback()
            raise

        return self._map_conversation(conversation)

    def get_conversation(self, conversation_id: str) -> ConversationResult:
        conversation = self._repository.get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(
                f"Conversation '{conversation_id}' was not found."
            )
        return self._map_conversation(conversation, include_messages=True)

    def list_conversations(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[ConversationResult, ...]:
        conversations = self._repository.list(limit=limit, offset=offset)
        return tuple(self._map_conversation(item) for item in conversations)

    def delete_conversation(self, conversation_id: str) -> None:
        try:
            deleted = self._repository.delete(conversation_id)
            if not deleted:
                self._repository.rollback()
                raise ConversationNotFoundError(
                    f"Conversation '{conversation_id}' was not found."
                )
            self._repository.commit()
        except ConversationNotFoundError:
            raise
        except Exception:
            self._repository.rollback()
            raise

    def query_conversation(
        self,
        *,
        conversation_id: str,
        request: RAGQueryRequest,
    ) -> ConversationQueryResult:
        conversation = self._repository.get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(
                f"Conversation '{conversation_id}' was not found."
            )

        history = tuple(
            ChatMessage(role=message.role, content=message.content)
            for message in conversation.messages[-self._history_limit :]
            if message.role in {"user", "assistant"}
        )

        enriched_request = RAGQueryRequest(
            question=request.question,
            limit=request.limit,
            document_ids=request.document_ids,
            allowed_version_ids=request.allowed_version_ids,
            minimum_relevance_score=request.minimum_relevance_score,
            deduplicate=request.deduplicate,
            conversation_history=history,
            generation_options=request.generation_options,
        )

        try:
            self._repository.add_message(
                conversation_id=conversation_id,
                role="user",
                content=request.question,
            )
            query_result = self._rag_query_service.query(enriched_request)
            self._repository.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=query_result.answer,
                citations=[
                    {
                        "source_number": citation.source_number,
                        "citation_text": citation.citation_text,
                        "chunk_id": citation.chunk_id,
                        "document_id": citation.document_id,
                        "version_id": citation.version_id,
                        "title": citation.title,
                        "page_number": citation.page_number,
                        "source_path": citation.source_path,
                        "excerpt": citation.excerpt,
                        "relevance_score": citation.relevance_score,
                        "metadata": citation.metadata,
                    }
                    for citation in query_result.citations
                ],
            )
            self._repository.commit()
        except Exception:
            self._repository.rollback()
            raise

        refreshed = self._repository.get(conversation_id)
        if refreshed is None:
            raise ConversationNotFoundError(
                f"Conversation '{conversation_id}' was not found after saving."
            )

        return ConversationQueryResult(
            conversation=self._map_conversation(
                refreshed,
                include_messages=True,
            ),
            query=query_result,
        )

    @staticmethod
    def _map_conversation(
        conversation,
        *,
        include_messages: bool = False,
    ) -> ConversationResult:
        messages = ()
        if include_messages:
            messages = tuple(
                ConversationMessageResult(
                    id=message.id,
                    role=message.role,
                    content=message.content,
                    citations=tuple(message.citations or []),
                    sequence_number=message.sequence_number,
                    created_at=message.created_at,
                )
                for message in conversation.messages
            )

        return ConversationResult(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=messages,
        )
